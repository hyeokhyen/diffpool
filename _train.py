
import sys
import warnings
warnings.filterwarnings("ignore") 
import torch.nn as nn
import time
import sklearn.metrics as metrics
import random
import gen.feat as featgen
import gen.data as datagen
import numpy as np
import torch
import encoders
from graph_sampler import GraphSampler
from torch.autograd import Variable

'''
Run.
Attach to encoder
Change to average pooling
'''

# syn_community1v2

input_dim = 10

n_range = range(40, 60)
m_range = range(4, 5)
num_graphs = 500
feature_generator = featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))
graphs1 = datagen.gen_ba(n_range, m_range, num_graphs, 
        feature_generator)
for G in graphs1:
  G.graph['label'] = 0

n_range = range(20, 30)
m_range = range(4, 5)
num_graphs = 500
inter_prob = 0.3
feature_generators = [featgen.ConstFeatureGen(np.ones(input_dim, dtype=float))]
graphs2 = datagen.gen_2community_ba(n_range, m_range, num_graphs, inter_prob, feature_generators)
for G in graphs2:
  G.graph['label'] = 1

graphs = graphs1 + graphs2

# prepare_data
random.shuffle(graphs)

train_ratio = 0.8
test_ratio = 0.1
train_idx = int(len(graphs) * train_ratio)
test_idx = int(len(graphs) * (1-test_ratio))
train_graphs = graphs[:train_idx]
val_graphs = graphs[train_idx: test_idx]
test_graphs = graphs[test_idx:]

print('Num training graphs: ', len(train_graphs), 
      '; Num validation graphs: ', len(val_graphs),
      '; Num testing graphs: ', len(test_graphs))

print('Number of graphs: ', len(graphs))
print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
print('Max, avg, std of graph size: ', 
        max([G.number_of_nodes() for G in graphs]), ', '
        "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])), ', '
        "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

# minibatch
feature_type = 'default'
max_nodes = 0
batch_size = 20
num_workers = 1
dataset_sampler = GraphSampler(train_graphs, 
                            normalize=False, 
                            max_num_nodes=max_nodes,
                            features=feature_type)
train_dataset_loader = torch.utils.data.DataLoader(
                          dataset_sampler, 
                          batch_size=batch_size, 
                          shuffle=True,
                          num_workers=num_workers)

dataset_sampler = GraphSampler(val_graphs, 
                        normalize=False,
                        max_num_nodes=max_nodes,
                        features=feature_type)
val_dataset_loader = torch.utils.data.DataLoader(
                        dataset_sampler, 
                        batch_size=batch_size, 
                        shuffle=False,
                        num_workers=num_workers)

dataset_sampler = GraphSampler(test_graphs, 
                        normalize=False,
                        max_num_nodes=max_nodes,
                        features=feature_type)
test_dataset_loader = torch.utils.data.DataLoader(
                        dataset_sampler, 
                        batch_size=batch_size, 
                        shuffle=False,
                        num_workers=num_workers)

max_num_nodes = dataset_sampler.max_num_nodes
assign_input_dim = dataset_sampler.assign_feat_dim

# soft-assign
hidden_dim = 20
output_dim = 20
num_classes = 2
num_gc_layers = 3
assign_ratio = 0.1
num_pool = 1
bn=True
linkpred=True
globalpoolaverage=True

model = encoders.SoftPoolingGcnEncoder(
        max_num_nodes, 
        input_dim, hidden_dim, 
        embedding_dim=output_dim, 
        label_dim=num_classes, 
        num_layers=num_gc_layers,
        assign_hidden_dim=hidden_dim, 
        assign_ratio=assign_ratio, 
        num_pooling=num_pool,
        bn=bn, 
        linkpred=linkpred, 
        assign_input_dim=assign_input_dim,
        globalpoolaverage=globalpoolaverage).cuda()

# train
num_epochs = 1000
mask_nodes = True
method = 'soft-assign'
clip=2.0
max_num_examples = 100
name = 'Train'

optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=0.001)

iter = 0
for epoch in range(num_epochs):
  total_time = 0
  avg_loss = 0.0

  # train
  model.train()
  print('Epoch: ', epoch)
  for batch_idx, data in enumerate(train_dataset_loader):
    begin_time = time.time()
    model.zero_grad()
    adj = Variable(data['adj'].float(), requires_grad=False).cuda()
    h0 = Variable(data['feats'].float(), requires_grad=False).cuda()
    label = Variable(data['label'].long()).cuda()
    batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None

    assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

    ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)

    print ('ypred:')
    print (ypred)
    print ('label:')
    print (label)
    assert False

    loss = model.loss(ypred, label, adj, batch_num_nodes)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    iter += 1
    avg_loss += loss
    #if iter % 20 == 0:
    #    print('Iter: ', iter, ', loss: ', loss.data[0])
    elapsed = time.time() - begin_time
    total_time += elapsed
  avg_loss /= batch_idx + 1

  print ('loss/avg_loss', avg_loss.item(), epoch)
  print ('loss/linkpred_loss', model.link_loss.item(), epoch)
  print ('epoch time: ', total_time)

  # evaluate
  labels = []
  preds = []
  for batch_idx, data in enumerate(train_dataset_loader):
      adj = Variable(data['adj'].float(), requires_grad=False).cuda()
      h0 = Variable(data['feats'].float()).cuda()
      labels.append(data['label'].long().numpy())
      batch_num_nodes = data['num_nodes'].int().numpy()
      assign_input = Variable(data['assign_feats'].float(), requires_grad=False).cuda()

      ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)
      _, indices = torch.max(ypred, 1)
      preds.append(indices.cpu().data.numpy())

      if max_num_examples is not None:
          if (batch_idx+1)*batch_size > max_num_examples:
              break

  labels = np.hstack(labels)
  preds = np.hstack(preds)
  
  result = {'prec': metrics.precision_score(labels, preds, average='macro'),
            'recall': metrics.recall_score(labels, preds, average='macro'),
            'acc': metrics.accuracy_score(labels, preds),
            'F1': metrics.f1_score(labels, preds, average="micro")}
  print(name, " accuracy:", result['acc'])
  print ('accuracy:', result['acc'])
  print ('---------------------------------')
