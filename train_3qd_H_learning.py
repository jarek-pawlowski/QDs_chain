import torch

import utils_nn
import utils


defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)

train_loader, test_loader = utils_nn.parse_dataset('./3qd_train/', './3qd_test/', batch_size=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = utils_nn.Autoencoder(parameters, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
run  = utils_nn.Experiments(device, model, train_loader, test_loader, optimizer)

train_loss, validation_loss = run.run_training(100)