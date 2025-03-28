import os
import numpy as np
import torch
import string
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
class RNNModel:
    """
    A character-level RNN model.
    """
    # make a bigger hidden size up to a thousand is ok
    # reduce learning rate, 10 -4 or -3
    def __init__(self, hidden_size=1000, seq_length=25, learning_rate=10e-7):
        # Hyperparameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Data and Mappings
        self.characters = []
        self.char2ix = {}
        self.ix2char = {}
        
        self.X_train = []
        self.y_train = []
        
        self.X_val = []
        self.y_val = []

        self.X_test = []
        self.y_test = []
        
        self.train_loss = []

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.hidden2output = None
        self.softmax = None

    def create_model(self):
        # made 42 ur hidden size
        # go up to 6 on num layers
        self.model = nn.Sequential(
            nn.Embedding(len(self.characters), self.hidden_size),
            nn.RNN(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=4, nonlinearity='tanh')
        )

        self.hidden2output = nn.Linear(self.hidden_size, len(self.characters))
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(
                        list(self.model.parameters()) + list(self.hidden2output.parameters()),
                        lr=self.learning_rate)

    def load_training_data(self):
        folder_path = 'train_data_mini'
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in txt_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                text_content = text_content.strip()
                text_content = text_content.replace('\n', '').replace('\r', '')
            for i in range(0, len(text_content), 30):
                token = text_content[i: i + 30]
                if len(token)  < 30:
                    break
                growing_x = []
                for char in token:
                    if (char not in self.characters):
                        self.characters.append(char)
                        self.char2ix[char] = len(self.characters) - 1
                        self.ix2char[len(self.characters) - 1] = char
                    growing_x.append(self.char2ix[char])
                self.X_train.append(growing_x)
                y_char = text_content[i + 30: i + 31]
                if (y_char not in self.characters):
                    self.characters.append(y_char)
                    self.char2ix[y_char] = len(self.characters) - 1
                    self.ix2char[len(self.characters) - 1] = y_char
                self.y_train.append(self.char2ix[y_char])
    
        self.characters.append("<unk>")
        self.char2ix["<unk>"] = len(self.characters) - 1
        self.ix2char[len(self.characters) - 1] = "<unk>"

        folder_path = 'val_data'
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        for file_name in txt_files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                text_content = text_content.strip()
                text_content = text_content.replace('\n', '').replace('\r', '')
            for i in range(0, len(text_content), 30):
                token = text_content[i: i + 30]
                if len(token)  < 30:
                    break
                growing_x = []
                for char in token:
                    if (char not in self.characters):
                        char = "<unk>"
                    growing_x.append(self.char2ix[char])
                self.X_val.append(growing_x)
                y_char = text_content[i + 30: i + 31]
                if (y_char not in self.characters):
                    y_char = "<unk>"
                self.y_val.append(self.char2ix[y_char])
            self.y_train = torch.tensor(self.y_train)
            self.X_train = torch.tensor(self.X_train)
            self.y_val = torch.tensor(self.y_val)
            self.X_val = torch.tensor(self.X_val)

    def load_test_data(self, test_file):
        with open(test_file, 'r', encoding='utf-8') as file:
            test_sequences = file.readlines()
        for sequence in test_sequences:
            sequence = sequence.strip()
            sequence = sequence.replace('\n', '').replace('\r', '')
            if len(sequence) < 30:
                sequence = ' ' * (30 - len(sequence)) + sequence
            elif len(sequence) > 30:
                sequence = sequence[-30:]

            input_data = []
            for char in sequence:
                if char not in self.characters:
                    char = "<unk>"
                input_data.append(self.char2ix[char])
            self.X_test.append(input_data)

        self.X_test = torch.tensor(self.X_test)

    def run_train(self, work_dir, num_epochs=100, batch_size=256):
        print (self.X_train.shape)
        val_losses = []
        self.model.to(device)
        self.X_train = self.X_train.to(device)
        self.y_train = self.y_train.to(device)
        self.X_val = self.X_val.to(device)
        self.y_val = self.y_val.to(device)
        self.hidden2output.to(device)
        for i in range(num_epochs):
            print("epoch: " + str(i))
            self.model.train()  # Set the model to training mode

            self.optimizer.zero_grad()
            epoch_loss = []
            for j in range(0, self.X_train.shape[0], batch_size):
                X_batch = self.X_train[j:j+batch_size]
                y_batch = self.y_train[j:j+batch_size]      
                y_pred, _ = self.model(X_batch)
                y_pred = y_pred[:, -1, :]
                y_pred = self.hidden2output(y_pred)
                loss = self.criterion(y_pred, y_batch) 
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            self.train_loss.append(np.mean(epoch_loss))
            print(np.mean(epoch_loss))
        # validation code!
        print(self.train_loss)
        self.model.eval()  
        with torch.no_grad():
            print(self.X_val.shape)
            val_pred, _ = self.model(self.X_val)
            val_pred = val_pred[:, -1, :]
            val_pred = self.hidden2output(val_pred)
            batch_loss = self.criterion(val_pred, self.y_val)
            val_losses.append(batch_loss.item())
            # print(batch_loss.item())
    def indices_to_string(self, indices_tensor):
        strings = []
        for indices in indices_tensor:
            chars = [
            self.ix2char.get(idx.item(), random.choice(list(self.ix2char.values())))
            for idx in indices
            ]
            chars = [
            self.ix2char[random.randint(0, len(self.characters) - 1)] if char == "<unk>" else char
            for char in chars
            ]
            string = ''.join(chars)
            strings.append(string)
        return strings

    def run_pred(self):
        self.X_test = self.X_test.to(device)
        self.model.to(device)
        self.hidden2output.to(device)
        self.softmax.to(device)
        
        y_test_pred, _ = self.model(self.X_test)
        y_test_pred = y_test_pred[:, -1, :]
        y_test_pred = self.hidden2output(y_test_pred)
        y_test_pred = self.softmax(y_test_pred)
        top_probs, top_indices = torch.topk(y_test_pred, 3, dim=1)
        pred_chars = self.indices_to_string(top_indices)
        print(pred_chars)
        return pred_chars

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def graph_loss(self, num_epochs=50):
        print("called")
        x = list(range(0, 50))
        plt.plot(x, self.train_loss)
        plt.title('Train loss vs epochs')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.show()


    def save(self, work_dir):
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

         # We need to check the format its being saved in
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hidden2output_state_dict': self.hidden2output.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'char2ix': self.char2ix,
            'ix2char': self.ix2char,
            'hidden_size': self.hidden_size,
            'learning_rate': self.learning_rate,
            'characters': self.characters,
        }, os.path.join(work_dir, 'model_ckpts/RNNmodel4.pth'))

    def load(self, work_dir):
        model_path = os.path.join(work_dir, 'src/model_ckpts/RNNmodel4.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the saved state
        if device == 'cuda':
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        self.characters = checkpoint['characters']
        self.char2ix = checkpoint['char2ix']
        self.ix2char = checkpoint['ix2char']
        self.hidden_size = checkpoint['hidden_size']
        self.learning_rate = checkpoint['learning_rate']
        
        model.create_model() #creates the model again

        #load_state_dict is a torch method that pulls params from the checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # so we need the hidden2output params saved seperate since it is its own layer
        self.hidden2output.load_state_dict(checkpoint['hidden2output_state_dict'])
        # helpful for if we want to resume training
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print(args.work_dir)
        print('Instantiating model')
        model = RNNModel()
        print('Loading training data')
        model.load_training_data()
        print('Creating the model...')
        model.create_model()
        print('Training')
        model.run_train(args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
        # print('Graphing losses')
        # model.graph_loss()
    elif args.mode == 'test':
      print('Loading model')
      model = RNNModel()
      model.load(args.work_dir)
      print(f'Loading test data from {args.test_data}')
      model.load_test_data(args.test_data)
      print('Making predictions')
      predictions = model.run_pred()
      print("writing predictions")
      RNNModel.write_pred(predictions, args.test_output)
