import engine.abs_engine as abs_engine
import torch
import torch.nn as nn
import numpy as np
import dataset.v2r_latent_set as v2r_set
import sklearn.metrics as metrics
import model.metric as v2r_metric
import torch.optim.lr_scheduler as lr_scheduler
import time
from tqdm import tqdm
import os
import json

import model.loss
import model.mm_infomax.model_latent as mminfomax_model


class InfoMax(abs_engine.AbsEngine):
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
        self.model = mminfomax_model.MMIM(hp=self.model_args, yaml_cfg=self.cfg["train"]).to(self.device)

        print("engine.mminfomax.InfoMax: audio mode = ", self.cfg["train"]["audio_mode"])

        mmilb_param = []
        main_param = []
        bert_param = []

        self.is_contrastive = self.cfg["train"].get("is_contrastive", True)
        self.add_va = self.cfg["train"].get("add_va", True)
        self.alpha = float(self.cfg["train"].get("alpha", 0.1))
        self.beta = float(self.cfg["train"].get("beta", 0.1))
        self.update_batch = int(self.cfg["train"].get("update_batch", 1))
        self.clip = float(self.cfg["train"].get("clip", 1.0))

        for name, p in self.model.named_parameters():
            # print(name)
            if p.requires_grad:
                if 'bert' in name:
                    bert_param.append(p)
                elif 'mi' in name:
                    mmilb_param.append(p)
                else: 
                    main_param.append(p)
                
            for p in (mmilb_param+main_param):
                if p.dim() > 1: # only tensor with no less than 2 dimensions are possible to calculate fan_in/fan_out
                    nn.init.xavier_normal_(p)
        print(f"engine.mminfomax.InfoMax: # mmilb param = {len(mmilb_param)}, # main param = {len(main_param)}")
        optim_mmilb_param = {"lr": float(self.cfg["train"]["lr_mmilb"])}
        if "weight_decay_mmilb" in self.cfg["train"]:
            optim_mmilb_param["weight_decay"] = float(self.cfg["train"].get("weight_decay_mmilb"))
        if "momentum_mmilb" in self.cfg["train"]:
            optim_mmilb_param["momentum"] = float(self.cfg["train"].get("momentum_mmilb"))
        
        self.optimizer_mmilb = getattr(torch.optim, self.cfg["train"].get("optim_mmilb", "SGD"))(mmilb_param, **optim_mmilb_param)

        # if len(bert_param) > 0:
        #     optimizer_main_group = [
        #         {'params': bert_param, 
        #          'weight_decay': float(self.cfg["train"].get("weight_decay_bert", 0.01)), 
        #          'lr': float(self.cfg["train"].get("lr_bert", 1e-4)),
        #         'momentum': float(self.cfg["train"].get("momentum_bert", 0.9))},
        #         {'params': main_param, 
        #          'weight_decay': float(self.cfg["train"].get("weight_decay_main", 0.01)), 
        #          'lr': float(self.cfg["train"].get("lr_main", 1e-4)),
        #          'momentum': float(self.cfg["train"].get("momentum_main", 0.9))
        #          }
        #     ]
        # else:
        #     main_group_main =  {'params': main_param, 
        #          'lr': float(self.cfg["train"].get("lr_main", 1e-4))
        #         }
        #     if "weight_decay_main" in self.cfg["train"]:
        #         main_group_main["weight_decay"] = float(self.cfg["train"].get("weight_decay_main"))
        #     if "momentum_main" in self.cfg["train"]:
        #         main_group_main["momentum"] = float(self.cfg["train"].get("momentum_main"))
            
        #     optimizer_main_group = [
        #        main_group_main
        #     ]

        # self.optimizer_main = getattr(torch.optim, self.cfg["train"].get("optim_mmilb", "SGD"))(
        #     optimizer_main_group
        # )

        main_group_main =  {
                'lr': float(self.cfg["train"].get("lr_main", 1e-4))
            }
        
        if "weight_decay_main" in self.cfg["train"]:
            main_group_main["weight_decay"] = float(self.cfg["train"].get("weight_decay_main"))
        if "momentum_main" in self.cfg["train"]:
            main_group_main["momentum"] = float(self.cfg["train"].get("momentum_main"))
            
        self.optimizer_main = getattr(torch.optim, self.cfg["train"].get("optim_main", "SGD"))(
            main_param, **main_group_main
        )


        self.scheduler_mmilb = lr_scheduler.StepLR(self.optimizer_mmilb, step_size=50, gamma=0.5)
        self.scheduler_main = lr_scheduler.StepLR(self.optimizer_main, step_size=50, gamma=0.5)

    def forward_pass(self, input_tuple, y=None, mem=None):
        visual = input_tuple["visual_feature"].to(self.device)
        audio_semantic = input_tuple["audio_semantic_feature"].to(self.device)
        audio_acoustic = input_tuple["audio_acoustic_feature"].to(self.device)
        clip_description = input_tuple["clip_description_embedding"].to(self.device)
        # label = input_tuple["reaction_distribution"].to(self.device)
        # pred, _ = self.model([visual, audio_semantic])
        # loss = nn.CrossEntropyLoss()(pred, label)
        # return loss, pred
        if self.cfg["train"]["audio_mode"] == "acoustic":
            audio = audio_acoustic
        elif self.cfg["train"]["audio_mode"] == "semantic":
            audio = audio_semantic
        elif self.cfg["train"]["audio_mode"] == "average":
            audio = (audio_acoustic + audio_semantic) / 2
        else:
            raise ValueError(f'audio mode {self.cfg["train"]["audio_mode"]} is not supported.')
        
        return self.model(visual, audio, clip_description, y, mem)  #  visual, acoustic, text, y=None, mem=None
    
    def train(self, dataloader, optim, criterion, stage):
        """
        replacing this part

        for batch_idx, input_tuple in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()

                # epoch_start_time = time.time()
                loss, _ = self.forward_pass_mmilb(input_tuple)
                loss.backward()

                self.optimizer_mmilb.step()
                train_losses_avg += loss.detach().cpu()

        """
        epoch_loss = 0
        mem_size = 1

        num_batches = len(dataloader)
        proc_loss, proc_size = 0, 0
        nce_loss = 0.0
        ba_loss = 0.0
        # start_time = time.time()

        left_batch = self.update_batch

        mem_pos_tv = []
        mem_neg_tv = []
        mem_pos_ta = []
        mem_neg_ta = []
        if self.add_va:
            mem_pos_va = []
            mem_neg_va = []

        num_train = 0

        for i_batch, batch_data in enumerate(dataloader):
            optim.zero_grad()

            y = batch_data["reaction_distribution"].to(self.device)  # label

            batch_size = y.size(0)

            num_train += batch_size

            if stage == 0:
                y = None
                mem = None
            elif stage == 1 and i_batch >= mem_size:
                mem = {'tv':{'pos':mem_pos_tv, 'neg':mem_neg_tv},
                        'ta':{'pos':mem_pos_ta, 'neg':mem_neg_ta},
                        'va': {'pos':mem_pos_va, 'neg':mem_neg_va} if self.add_va else None}
            else:
                mem = {'tv': None, 'ta': None, 'va': None}

            lld, nce, preds, pn_dic, H = self.forward_pass(batch_data, y=y, mem=mem)

            if stage == 1:
                y_loss = criterion(preds, y)
                
                if len(mem_pos_tv) < mem_size:
                    mem_pos_tv.append(pn_dic['tv']['pos'].detach())
                    mem_neg_tv.append(pn_dic['tv']['neg'].detach())
                    mem_pos_ta.append(pn_dic['ta']['pos'].detach())
                    mem_neg_ta.append(pn_dic['ta']['neg'].detach())
                    if self.add_va:
                        mem_pos_va.append(pn_dic['va']['pos'].detach())
                        mem_neg_va.append(pn_dic['va']['neg'].detach())
                
                else: # memory is full! replace the oldest with the newest data
                    oldest = i_batch % mem_size
                    mem_pos_tv[oldest] = pn_dic['tv']['pos'].detach()
                    mem_neg_tv[oldest] = pn_dic['tv']['neg'].detach()
                    mem_pos_ta[oldest] = pn_dic['ta']['pos'].detach()
                    mem_neg_ta[oldest] = pn_dic['ta']['neg'].detach()

                    if self.add_va:
                        mem_pos_va[oldest] = pn_dic['va']['pos'].detach()
                        mem_neg_va[oldest] = pn_dic['va']['neg'].detach()

                if self.is_contrastive:
                    loss = y_loss + self.alpha * nce - self.beta * lld
                else:
                    loss = y_loss
                if i_batch > mem_size:
                    loss -= self.beta * H
                loss.backward()
                print(f"stage1: ce_loss = {y_loss.item()}, nce_loss = {nce.item()}, lld_loss = {-lld.item()}, H = {H}, total_loss = {loss.item()}")
                
            elif stage == 0:
                # maximize likelihood equals minimize neg-likelihood
                loss = -lld
                loss.backward()
                print(f"stage0: lld_loss = {-lld.item()}, total loss = {loss.item()}")
            else:
                raise ValueError('stage index can either be 0 or 1')
            
            left_batch -= 1
            if left_batch == 0:
                left_batch = self.update_batch
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                optim.step()
            
            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            nce_loss += nce.item() * batch_size
            ba_loss += (-H - lld) * batch_size

            # if i_batch % self.hp.log_interval == 0 and i_batch > 0:
            #     avg_loss = proc_loss / proc_size
            #     elapsed_time = time.time() - start_time
            #     avg_nce = nce_loss / proc_size
            #     avg_ba = ba_loss / proc_size
            #     print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss ({}) {:5.4f} | NCE {:.3f} | BA {:.4f}'.
            #         format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hp.log_interval, 'TASK+BA+CPC' if stage == 1 else 'Neg-lld',
            #         avg_loss, avg_nce, avg_ba))
            #     proc_loss, proc_size = 0, 0
            #     nce_loss = 0.0
            #     ba_loss = 0.0
            #     start_time = time.time()
                
        return epoch_loss / len(dataloader)

    def run(self):
        cfg_train =  self.cfg['train']
        num_epochs = cfg_train['num_epochs']

        time_records = []  # key: epoch start time, epoch end time, train time, val time 

        train_losses_mmilb  = []
        train_losses_main  = []

        train_performances = []
        val_performances = []
        test_performances = []

        val_gts = []
        val_preds = []
        test_gts = []
        test_preds = []

        best_val_loss = 1e5  # some really big number 
        best_epoch = -1

        for epoch in tqdm(range(num_epochs)):
            time_record = {}
            self.set_train()

            # start recording training time
            train_start_time = time.time()
            time_record["train_start_time"] = train_start_time
            time_record["train_epoch_time"] = []

            # record training performance
            # train_losses_avg = 0

            # train the models
            # stage 0
           
                # epoch_end_time = time.time()
                # print(f"batch {batch_idx}, training time {epoch_end_time - epoch_start_time}", flush=True)
                # time_record["train_epoch_time"].append(epoch_end_time - epoch_start_time)
            
            # train_losses_avg /= len(self.train_loader)

            loss_fn = getattr(model.loss, self.cfg["train"].get("loss_fn", "CrossEntropyLoss"))()

            train_loss_mmilb = self.train(self.train_loader, self.optimizer_mmilb, loss_fn, stage=0)
            train_losses_mmilb.append(train_loss_mmilb)

            # record training performance
            # train_losses_avg = 0
            # train the models
            # stage 1
            # for batch_idx, input_tuple in enumerate(self.train_loader):
                
            #     self.optimizer.zero_grad()

            #     # epoch_start_time = time.time()
            #     loss, _ = self.forward_pass(input_tuple)
            #     loss.backward()

            #     self.optimizer_main.step()
            #     train_losses_avg += loss.detach().cpu()
            
            # train_losses_avg /= len(self.train_loader)
            train_loss_main = self.train(self.train_loader, self.optimizer_main, loss_fn, stage=1)
            train_losses_main.append(train_loss_main)
            
            # record training time
            train_end_time = time.time()
            time_record["train_end_time"] = train_end_time
            time_record["train_time"] = train_end_time - train_start_time

            # record validation performance
            # start recording validation time
            val_start_time = time.time()
            time_record["val_start_time"] = val_start_time

            self.set_eval()

            with torch.no_grad():
                # validate the models                
                
                if self.test_mod:
                    test_performance_avg = self.evaluate(self.test_loader)
                    test_performances.append(test_performance_avg)
                
                train_performance_avg = self.evaluate(self.train_loader)
                val_performance_avg = self.evaluate(self.val_loader)

                val_pred, val_gt = self.inference(self.val_loader)
                test_pred, test_gt = self.inference(self.test_loader)
                
                val_gts.append(val_gt)
                val_preds.append(val_pred)

                test_gts.append(test_gt)
                test_preds.append(test_pred)
            
            np.save(os.path.join(self.report_path, "val_preds.npy"), np.stack(val_preds, axis=0))
            np.save(os.path.join(self.report_path, "val_gts.npy"), np.stack(val_gts, axis=0))
            np.save(os.path.join(self.report_path, "test_preds.npy"), np.stack(test_preds, axis=0))
            np.save(os.path.join(self.report_path, "test_gts.npy"), np.stack(test_gts, axis=0))

            # record validation time
            val_end_time = time.time()
            time_record["val_end_time"] = val_end_time
            time_record["val_time"] = val_end_time - val_start_time 

            time_records.append(time_record)

            train_performances.append(train_performance_avg)
            val_performances.append(val_performance_avg)

            val_loss_avg = val_performance_avg["loss"]

            # save the best models
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(
                    {
                        "model": self.model.state_dict(), 
                        "epoch": epoch,
                        "optimizer_main": self.optimizer_main.state_dict(),
                        "optimizer_mmilb": self.optimizer_mmilb.state_dict(),
                        "val_loss": best_val_loss,
                        "val_performance": val_performance_avg
                    }, 
                    os.path.join(self.report_path, f"model.pth.tar"))
                print(f"Saved model at epoch {epoch} as model.pth.tar")
                best_epoch = epoch
                
            # dump the results:
            with open(os.path.join(self.report_path, "time_records.json"), "w") as f:
                json.dump(time_records, f, indent=4)
            
            json_hist = []
            if not self.test_mod:
                for idx, (train_loss_mmilb, train_loss_main, train_performance, val_performance) in enumerate(zip(train_losses_mmilb, train_losses_main, train_performances, val_performances)):
                    json_hist.append({
                        "epoch": idx,
                        "train_loss_mmilb": train_loss_mmilb,
                        "train_loss_main": train_loss_main,
                        "train_performance": train_performance,
                        "val_performance": val_performance,
                        "best_epoch": best_epoch
                    })
            else:
                for idx, (train_loss_mmilb, train_loss_main, train_performance, val_performance, test_performance) in enumerate(zip(train_losses_mmilb, train_losses_main, train_performances, val_performances, test_performances)):
                    json_hist.append({
                        "epoch": idx,
                        "train_loss_mmilb": train_loss_mmilb,
                        "train_loss_main": train_loss_main,
                        "train_performance": train_performance,
                        "val_performance": val_performance,
                        "test_performance": test_performance,
                        "best_epoch": best_epoch
                    })
            with open(os.path.join(self.report_path, "performance.json"), "w") as f:
                json.dump(json_hist, f, indent=4)

            self.scheduler_mmilb.step()
            self.scheduler_main.step()


    def evaluate(self, dataloader):
        """
        just compute accuracy for now
        """
        self.set_eval()
        gt_labels = []
        pred_labels = []

        loss = nn.CrossEntropyLoss()

        loss_avg = 0

        for input_tuple in dataloader:
            out = self.forward_pass(input_tuple)
            pred = out[2]

            loss_avg += loss(pred, input_tuple["reaction_distribution"].to(self.device)).item()

            gt_labels.append(input_tuple["reaction_distribution"].detach().cpu().numpy())
            pred_labels.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
            # pred_labels.append(pred.detach().cpu().numpy())
        
        gt_labels = np.concatenate(gt_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)

        loss_avg /= len(dataloader)

        return {
            "ldl": v2r_metric.compute_all_distribution_metrics(pred_labels, gt_labels, output_float=True),
            # "pred_dist": pred_labels,
            "loss": loss_avg,
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
            out = self.forward_pass(input_tuple)
            pred = out[2]
            gt_labels.append(input_tuple["reaction_distribution"].detach().cpu().numpy())
            pred_labels.append(torch.softmax(pred, dim=1).detach().cpu().numpy())
            # pred_labels.append(pred.detach().cpu().numpy())
        
        gt_labels = np.concatenate(gt_labels, axis=0)
        pred_labels = np.concatenate(pred_labels, axis=0)

        return pred_labels, gt_labels