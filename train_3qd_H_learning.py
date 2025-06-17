import torch

import utils_nn
import utils


defaults = utils.Defaults()
parameters = utils.Parameters(no_dots=3, no_levels=1, default_parameters=defaults)

#train_loader, test_loader = utils_nn.parse_dataset('./3qd_train/', './3qd_test/', batch_size=10)
train_loader, test_loader = utils_nn.parse_dataset('./3qd_train1/', './3qd_train/', batch_size=10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = utils_nn.Autoencoder(parameters, device, bypass=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

run  = utils_nn.Experiments(device, model, train_loader, test_loader, optimizer, bypass=True)
train_loss, validation_loss = run.run_training(100)
utils_nn.plot_loss(train_loss, validation_loss)

pr_map, ref_map = run.get_prediction(0)
plot = utils.Plotting(directory='./tests')

plot.plot_conductance_map_(pr_map[0], suffix='prediction_0')
plot.plot_conductance_map_(ref_map[0], suffix='gt_0')

plot.plot_conductance_map_(pr_map[1], suffix='prediction_1')
plot.plot_conductance_map_(ref_map[1], suffix='gt_1')

plot.plot_conductance_map_(pr_map[-1], suffix='prediction_15')
plot.plot_conductance_map_(ref_map[-1], suffix='gt_15')