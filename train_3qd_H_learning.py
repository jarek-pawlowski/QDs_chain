import torch

import utils_nn
import utils


defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)

train_loader, test_loader = utils_nn.parse_dataset('./3qd_train1/', './3qd_test/', batch_size=10)
#train_loader, test_loader = utils_nn.parse_dataset('./3qd_train0/', './3qd_train0/', batch_size=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = utils_nn.Autoencoder(parameters, device, bypass=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

run  = utils_nn.Experiments(device, model, train_loader, test_loader, optimizer, bypass=True)
train_loss, validation_loss = run.run_training(2000)

pr_map, ref_map = run.get_prediction(0)
plot = utils.Plotting(directory='./tests')

plot.plot_conductance_map_(pr_map[0], suffix='prediction')
plot.plot_conductance_map_(ref_map[0], suffix='gt')