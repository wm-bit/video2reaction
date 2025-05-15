import os
import json

import torch
import torch.nn as nn
import copy
import time
from tqdm import tqdm
import itertools
from model.device_check import *
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler


class AbsEngine:
    def __init__(self, cfg, report_path, **kwargs):
        self.cfg = cfg
        self.report_path = report_path
        self.test_mod = kwargs.get("test_mod", False)
        self.model_args = kwargs.get("model_args", None)
        self.device = device

        self.train_set = None
        self.test_set = None
        self.val_set = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
        self.init_data()

        self.model = None
        self.optimizer = None
        self.init_model()
        if self.optimizer is not None:
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

    def init_data(self):
        pass

    def init_model(self):
        pass

    def forward_pass(self, input_tuple):
        """
        return:
            losses: 1D tensor, shape: (num_models,), loss for each model
        """
        pass

    def evaluate(self, input_tuple):
        """
        return:
            mi_est: 1D tensor, shape: (num_models,), mi estimate for each model
        """
        pass

    def inference(self, dataloader):
        pass

    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()

    def run(self):
        cfg_train =  self.cfg['train']
        num_epochs = cfg_train['num_epochs']

        time_records = []  # key: epoch start time, epoch end time, train time, val time 
        train_losses  = []
        train_performances = []
        val_losses = []
        val_performances = []
        test_losses = []
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

            # record training performance
            train_losses_avg = 0
            train_performance_avg = 0

            # start recording training time
            train_start_time = time.time()
            time_record["train_start_time"] = train_start_time
            time_record["train_epoch_time"] = []

            # train the models
            for batch_idx, input_tuple in enumerate(self.train_loader):
                
                self.optimizer.zero_grad()

                # epoch_start_time = time.time()
                loss, _ = self.forward_pass(input_tuple)
                loss.backward()

                self.optimizer.step()
                train_losses_avg += loss.detach().cpu()
                # epoch_end_time = time.time()
                # print(f"batch {batch_idx}, training time {epoch_end_time - epoch_start_time}", flush=True)
                # time_record["train_epoch_time"].append(epoch_end_time - epoch_start_time)
            
            train_losses_avg /= len(self.train_loader)
            
            # record training time
            train_end_time = time.time()
            time_record["train_end_time"] = train_end_time
            time_record["train_time"] = train_end_time - train_start_time

            # record validation performance
            val_losses_avg = 0
            val_performance_avg = 0

            test_losses_avg = 0

            # start recording validation time
            val_start_time = time.time()
            time_record["val_start_time"] = val_start_time

            self.set_eval()

            with torch.no_grad():
                # validate the models                
                for input_tuple in self.val_loader:
                    loss, _ = self.forward_pass(input_tuple)
                    val_losses_avg += loss.detach().cpu()
                
                if self.test_mod:
                    for input_tuple in self.test_loader:
                        loss, _ = self.forward_pass(input_tuple)
                        test_losses_avg += loss.detach().cpu()
                    test_losses_avg /= len(self.test_loader)
                    test_performance_avg = self.evaluate(self.test_loader)
                    test_performances.append(test_performance_avg)
                    test_losses.append(test_losses_avg)
                
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
            
            val_losses_avg /= len(self.val_loader)

            # record validation time
            val_end_time = time.time()
            time_record["val_end_time"] = val_end_time
            time_record["val_time"] = val_end_time - val_start_time 

            time_records.append(time_record)

            train_losses.append(train_losses_avg)
            train_performances.append(train_performance_avg)
            val_losses.append(val_losses_avg)
            val_performances.append(val_performance_avg)

            # save the best models
            if val_losses_avg < best_val_loss:
                best_val_loss = val_losses_avg
                torch.save(
                    {
                        "model": self.model.state_dict(), 
                        "epoch": epoch,
                        "optimizer": self.optimizer.state_dict(),
                        "val_loss": best_val_loss.item(),
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
                for idx, (train_loss, train_performance, val_loss, val_performance) in enumerate(zip(train_losses, train_performances, val_losses, val_performances)):
                    json_hist.append({
                        "epoch": idx,
                        "train_loss": train_loss.item(),
                        "train_performance": train_performance,
                        "val_loss": val_loss.item(),
                        "val_performance": val_performance,
                        "best_epoch": best_epoch
                    })
            else:
                for idx, (train_loss, train_performance, val_loss, val_performance, test_loss, test_performance) in enumerate(zip(train_losses, train_performances, val_losses, val_performances, test_losses, test_performances)):
                    json_hist.append({
                        "epoch": idx,
                        "train_loss": train_loss.item(),
                        "train_performance": train_performance,
                        "val_loss": val_loss.item(),
                        "val_performance": val_performance,
                        "test_loss": test_loss.item(),
                        "test_performance": test_performance,
                        "best_epoch": best_epoch
                    })
            with open(os.path.join(self.report_path, "performance.json"), "w") as f:
                json.dump(json_hist, f, indent=4)

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()
    
    def test(self):
        print("load trained model to test")
        cp = torch.load(os.path.join(self.report_path, f"model.pth.tar"))
        self.model.load_state_dict(cp["model"])
        self.set_eval()
        with torch.no_grad():
            test_performance_avg = self.evaluate(self.test_loader)

        print(f"test accuracy at epoch {cp['epoch']} : {test_performance_avg}")

        with open(os.path.join(self.report_path, "test_performance.json"), "w") as f:
            json.dump({"test_acc": test_performance_avg, "epoch": cp["epoch"], 
                       "val_loss": cp["val_loss"], "val_performance": cp["val_performance"]}, 
                       f, 
                       indent=4)
        
        test_pred, test_gt = self.inference(self.test_loader)
        np.save(os.path.join(self.report_path, "final_test_pred.npy"), test_pred)
        np.save(os.path.join(self.report_path, "final_test_gt.npy"), test_gt)
        
        return test_performance_avg, cp["val_performance"]
