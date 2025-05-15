import engine.abs_engine as abs_engine
import torch
import torch.nn as nn
import numpy as np
import dataset.v2r_latent_set as v2r_set
import model.cten_vaanet_erase as cten_vaanet_erase
import sklearn.metrics as metrics
import model.metric as v2r_metric
import model.loss


class VAANetErase(abs_engine.AbsEngine):
    def init_data(self):
        self.train_set = v2r_set.Video2Reaction(metadata_file_path=f"/project/youtube_video/metadata/video2reaction-full/train.json",
                                processed_feature_dir="/project/youtube_video/processed_features/",
                                input_type="processed",
                                device="cuda",
                                lazy_load=False,
                                use_time_dimension=False,
                                key_frame_dir="/project/youtube_video/key_frames/",
                                cache_file_path=f"/scratch3/workspace/reaction-video-dataset/cache/video2reaction-full/train_vit_bert-base-uncased_clap_general_hubert_large.pt",
                                sample_size=None)
        
        self.val_set = v2r_set.Video2Reaction(metadata_file_path=f"/project/youtube_video/metadata/video2reaction-full/val.json",
                                processed_feature_dir="/project/youtube_video/processed_features/",
                                input_type="processed",
                                device="cuda",
                                lazy_load=False,
                                use_time_dimension=False,
                                key_frame_dir="/project/youtube_video/key_frames/",
                                cache_file_path=f"/scratch3/workspace/reaction-video-dataset/cache/video2reaction-full/val_vit_bert-base-uncased_clap_general_hubert_large.pt",
                                sample_size=None)
        
        self.test_set = v2r_set.Video2Reaction(metadata_file_path=f"/project/youtube_video/metadata/video2reaction-full/test.json",
                                processed_feature_dir="/project/youtube_video/processed_features/",
                                input_type="processed",
                                device="cuda",
                                lazy_load=False,
                                use_time_dimension=False,
                                key_frame_dir="/project/youtube_video/key_frames/",
                                cache_file_path=f"/scratch3/workspace/reaction-video-dataset/cache/video2reaction-full/test_vit_bert-base-uncased_clap_general_hubert_large.pt",
                                sample_size=None)
        
        batch_size = self.cfg["train"]["batch_size"]
        
        self.train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size=batch_size,
                                                        shuffle=True
                                                        )

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
                                                        batch_size=batch_size
                                                        )

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                        batch_size=batch_size
                                                        )


    def init_model(self):
        self.model = cten_vaanet_erase.VAANetErase(n_classes=21, seq_len=16).to(self.device)
        print("engine.cten.VAANetErease: audio mode = ", self.cfg["train"]["audio_mode"])
        print("engine.cten.VAANetErease: loss func = ", self.cfg["train"].get("loss_fn", "CrossEntropyLoss"))

        self.is_erasing = self.cfg["train"].get("is_erasing", False)
        print("engine.cten.VAANetErease: is_erasing = ", self.is_erasing)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                          lr=self.cfg["train"]["lr"], 
                                          weight_decay=float(self.cfg["train"].get("weight_decay", 0.01)),
                                          momentum=float(self.cfg["train"].get("momentum", 0.9)))

    def forward_pass(self, input_tuple):
        loss_func = getattr(model.loss, self.cfg["train"].get("loss_fn", "CrossEntropyLoss"))()

        visual = input_tuple["visual_feature"].to(self.device)
        audio_semantic = input_tuple["audio_semantic_feature"].to(self.device)
        audio_acoustic = input_tuple["audio_acoustic_feature"].to(self.device)

        if self.cfg["train"]["audio_mode"] == "acoustic":
            audio = audio_acoustic
        elif self.cfg["train"]["audio_mode"] == "semantic":
            audio = audio_semantic
        elif self.cfg["train"]["audio_mode"] == "average":
            audio = (audio_acoustic + audio_semantic) / 2
        else:
            raise ValueError(f'audio mode {self.cfg["train"]["audio_mode"]} is not supported.')

        label = input_tuple["reaction_distribution"].to(self.device)
        pred, gamma1 = self.model([visual, audio])
        # pred = torch.softmax(pred, dim=1)
        # loss = nn.CrossEntropyLoss()(pred, label)
        loss = loss_func(pred, label)
        
        if not self.is_erasing or not torch.is_grad_enabled():
            # print("no erase")
            return loss, pred
        else:
            gamma_row_max=torch.max(gamma1,dim=1)[0]*0.7 + torch.min(gamma1,dim=1)[0]*0.3
            gamma_row_max=gamma_row_max.unsqueeze(0).transpose(1,0)
            gamma_thre=gamma_row_max.expand(gamma1.shape)
            high_index=gamma1<gamma_thre
            low_index=gamma1>gamma_thre


            pred2, gamma2 = self.model([visual * high_index.unsqueeze(2), audio * high_index.unsqueeze(2)])
            loss2 = loss_func(pred2, label)

            pred3, gamma3 = self.model([visual * low_index.unsqueeze(2), audio * low_index.unsqueeze(2)])
            loss3 = loss_func(pred3, label)

            return (loss + loss2 + loss3) / 3, pred


    def evaluate(self, dataloader):
        """
        just compute accuracy for now
        """
        self.set_eval()
        gt_labels = []
        pred_labels = []
        for input_tuple in dataloader:
            _, pred = self.forward_pass(input_tuple)
            gt_labels.append(input_tuple["reaction_distribution"].detach().cpu().numpy())
            pred_labels.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
            # pred_labels.append(pred.detach().cpu().numpy())
        
        gt_labels = np.concatenate(gt_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)

        return {
            "ldl": v2r_metric.compute_all_distribution_metrics(pred_labels, gt_labels, output_float=True),
            # "pred_dist": pred_labels,
            "cls": v2r_metric.compute_all_classification_metrics(pred_labels, gt_labels, output_float=True)
        }
    

    def inference(self, dataloader):
        """
        just compute accuracy for now
        """
        self.set_eval()
        gt_labels = []
        pred_labels = []
        for input_tuple in dataloader:
            _, pred = self.forward_pass(input_tuple)
            gt_labels.append(input_tuple["reaction_distribution"].detach().cpu().numpy())
            pred_labels.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
            # pred_labels.append(pred.detach().cpu().numpy())
        
        gt_labels = np.concatenate(gt_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)

        return pred_labels, gt_labels