import torch.nn as nn 
import torch.nn.functional as F
#add train
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20 * 24, 5000) #fully connected layer 1 
        self.bn1 = nn.BatchNorm1d(5000) #batchnorm layer 1 
        self.fc2 = nn.Linear(5000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.res_fc1 = nn.Linear(1000, 1000) #residual layers
        self.res_bn1 = nn.BatchNorm1d(1000) #all 1000 bc after
        self.res_fc2 = nn.Linear(1000, 1000)
        self.res_bn2 = nn.BatchNorm1d(1000)
        self.fc_out = nn.Linear(1000, 1)
        
    def forward(self, x):
        x = x.float()
        x = x.view(-1, 20 * 24)
        x = self.fc1(x)
        #anacx = self.bn1(x)
        x = F.relu(x) 
        x = self.fc2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        # resnet blocks
        for block_num in range(4):
            res_inp = x
            x = self.res_fc1(x)
            #x = self.res_bn1(x)
            x = F.relu(x)
            x = self.res_fc2(x)
            #x = self.res_bn2(x)
            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x

    def train_nnet(self, x):
        return x 

"""
def train_nnet(nnet: nn.Module, states_nnet: List[np.ndarray], outputs: np.ndarray, device: torch.device,
               batch_size: int, num_itrs: int, train_itr: int, lr: float, lr_d: float, display: bool = True) -> float:
    # optimization
    display_itrs = 100
    criterion = nn.MSELoss()
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # initialize status tracking
    start_time = time.time()

    # train network
    batches: List[Tuple[List, np.ndarray]] = make_batches(states_nnet, outputs, batch_size)

    nnet.train()
    max_itrs: int = train_itr + num_itrs

    last_loss: float = np.inf
    batch_idx: int = 0
    while train_itr < max_itrs:
        # zero the parameter gradients
        optimizer.zero_grad()
        lr_itr: float = lr * (lr_d ** train_itr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_itr

        # get data
        inputs_batch, targets_batch_np = batches[batch_idx]
        targets_batch_np = targets_batch_np.astype(np.float32)

        # send data to device
        states_batch: List[Tensor] = states_nnet_to_pytorch_input(inputs_batch, device)
        targets_batch: Tensor = torch.tensor(targets_batch_np, device=device)

        # forward
        nnet_outputs_batch: Tensor = nnet(*states_batch)

        # cost
        nnet_cost_to_go = nnet_outputs_batch[:, 0]
        target_cost_to_go = targets_batch[:, 0]

        loss = criterion(nnet_cost_to_go, target_cost_to_go)

        # backwards
        loss.backward()

        # step
        optimizer.step()

        last_loss = loss.item()
        # display progress
        if (train_itr % display_itrs == 0) and display:
            print("Itr: %i, lr: %.2E, loss: %.2f, targ_ctg: %.2f, nnet_ctg: %.2f, "
                  "Time: %.2f" % (
                      train_itr, lr_itr, loss.item(), target_cost_to_go.mean().item(), nnet_cost_to_go.mean().item(),
                      time.time() - start_time))

            start_time = time.time()

        train_itr = train_itr + 1

        batch_idx += 1
        if batch_idx >= len(batches):
            shuffle(batches)
            batch_idx = 0

    return last_loss


"""
