# import torch
# import click
#
# from data import mnist
# from src.models.model import MyAwesomeModel
#
#
# @click.group()
# def cli():
#     pass
#
#
# @click.command()
# @click.option("--lr", default=1e-3, help='learning rate to use for training')
# def train(lr):
#     print("Training day and night")
#     print(lr)
#
#     model = MyAwesomeModel()
#     train_set, _ = mnist()
#
#
#     model = MyAwesomeModel()
#     model = model.to(self.device)
#     train_set = CorruptMnist(train=True)
#     dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     n_epoch = 5
#     for epoch in range(n_epoch):
#         loss_tracker = []
#         for batch in dataloader:
#             optimizer.zero_grad()
#             x, y = batch
#             preds = model(x.to(self.device))
#             loss = criterion(preds, y.to(self.device))
#             loss.backward()
#             optimizer.step()
#             loss_tracker.append(loss.item())
#         print(f"Epoch {epoch + 1}/{n_epoch}. Loss: {loss}")
#     torch.save(model.state_dict(), 'trained_model.pt')
#
#     plt.plot(loss_tracker, '-')
#     plt.xlabel('Training step')
#     plt.ylabel('Training loss')
#     plt.savefig("training_curve.png")
#
#     return model
#
#
# @click.command()
# @click.argument("model_checkpoint")
# def evaluate(model_checkpoint):
#     print("Evaluating until hitting the ceiling")
#     print(model_checkpoint)
#
#     model = torch.load(model_checkpoint)
#     _, test_set = mnist()
#
#
# cli.add_command(train)
# cli.add_command(evaluate)
#
#
# if __name__ == "__main__":
#     cli()
