import pandas as pd
import pysmiles
import networkx as nx
import torch
import torch_geometric as pyg
import numpy as np
import os
import logging
from torch.utils.data import Dataset, DataLoader
#from torch_geometric.data import Batch, Dataset
#from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import tqdm
import networkx as nx
import matplotlib.pyplot as plt

BREAK = '**************************************************'

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


DIR = 'molecules'
READOUT = 'MIN'

##############################################################
# NETWORK DEFINITIONS
class SAGEMolGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(SAGEConv(in_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        #x = torch.mean(x, dim=0)
        #x = torch.sum(x, dim=0)
        #x = torch.max(x, dim=0).values
        x = torch.min(x, dim=0).values
        return x

# define knowledge graph GNN
class SAGEKGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(SAGEKGCN, self).__init__()
        self.conv1 = GCNConv(out_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.mol_gnn = SAGEMolGCN(in_channels, out_channels, heads)

    def forward(self, x, edge_index):
        outs = []
        for mol in x:
          outs.append(self.mol_gnn(mol.x, mol.edge_index))
        x = torch.stack(outs, dim=0)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GATMolGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        #x = torch.sum(x, dim=0)
        #x = torch.mean(x, dim=0)
        #x = torch.max(x, dim=0).values
        x = torch.min(x, dim=0).values
        return x

# define knowledge graph GNN
class GATKGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, heads):
        super(GATKGCN, self).__init__()
        self.conv1 = GCNConv(out_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.mol_gnn = GATMolGCN(in_channels, out_channels, hidden_channels, heads)

    def forward(self, x, edge_index):
        outs = []
        for mol in x:
          outs.append(self.mol_gnn(mol.x, mol.edge_index))
        x = torch.stack(outs, dim=0)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GCNMolGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNMolGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, 2*out_channels)
        self.conv3 = GCNConv(2*out_channels, out_channels)
    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index).relu()
        out = self.conv2(out, edge_index).relu()
        out = self.conv3(out, edge_index)
        #out = torch.sum(out, axis=0)
        #out = torch.mean(out, dim=0)
        #out = torch.max(out, dim=0).values
        out = torch.min(out, dim=0).values
        return out

# define knowledge graph GNN
class GCNKGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNKGCN, self).__init__()
        self.conv1 = GCNConv(out_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)
        self.mol_gnn = GCNMolGCN(in_channels, out_channels)

    def forward(self, x, edge_index):
        outs = []
        for mol in x:
          outs.append(self.mol_gnn(mol.x, mol.edge_index))
        x = torch.stack(outs, dim=0)
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

################################################################

################################################################
# KNOWLEDGE GRAPH SET UP 
################################################################
df_smiles = pd.read_csv('chem_smiles.csv')

all_possible_nodes = set()
for i in range(df_smiles.shape[0]):
  g = pysmiles.read_smiles(df_smiles.iloc[i].SMILES)
  node_feature_set = set(list(nx.get_node_attributes(g, name='element')
                                                                    .values()))
  for el in node_feature_set:
    all_possible_nodes.add(el)

# Apply embedding from pytorch (i.e. for embedding in transformers)
nu_atoms = len(all_possible_nodes)
embeds = torch.nn.Embedding(nu_atoms, 32)
atom_dict = {}
for i, j in enumerate(all_possible_nodes):
  atom_dict[j] = embeds(torch.tensor(i, dtype=torch.long))

# Apply one-hot encoding for each atom
atom_one_hot = {}
for i, j in enumerate(all_possible_nodes):
  one_hot = np.zeros(nu_atoms)
  one_hot[i] = 1
  atom_one_hot[j] = torch.tensor(one_hot, dtype=torch.long)


compound_graph = nx.read_graphml('comp-comp-by-ko-by-rn-attr')

logging.info('Making molecule graphs...')

try:
    os.mkdir(DIR)
except:
    logging.info('{} already exists!'.format(DIR))

pubchem_ids = []
graph_file_names = []
smile_strings = []
for i in range(df_smiles.shape[0]):
  g = pysmiles.read_smiles(df_smiles.iloc[i].SMILES)
  smile_strings.append(df_smiles.iloc[i].SMILES)
  pubchem_id = df_smiles.iloc[i].ID
  for j in range(len(g.nodes)):
    g.nodes[j]['el_one_hot'] = atom_one_hot[g.nodes[j]['element']]
    g.nodes[j]['el_word_vec'] = atom_dict[g.nodes[j]['element']]
  graph_file_name = 'graph_{}'.format(pubchem_id)
  nx.write_gpickle(g, 'molecules/' + graph_file_name)
  pubchem_ids.append(pubchem_id)
  graph_file_names.append(graph_file_name)


# Uncomment this if you don't already have the saved dataframe!
#graph_df = pd.DataFrame({'ID': pubchem_ids, 'graph_file': graph_file_names})
#graph_df.to_csv('molecule_graph_df.csv')

for nodes in compound_graph.nodes:
  if 'graph_file' not in compound_graph.nodes[nodes]:
    smiles = compound_graph.nodes[nodes]['smiles']
    smile_strings.append(smiles)
    pubchem_id = nodes
    g = pysmiles.read_smiles(smiles)
    for j in range(len(g.nodes)):
      g.nodes[j]['el_one_hot'] = atom_one_hot[g.nodes[j]['element']]
      g.nodes[j]['el_word_vec'] = atom_dict[g.nodes[j]['element']]
    graph_file_name = 'graph_{}'.format(pubchem_id)
    nx.write_gpickle(g, 'molecules/' + graph_file_name)
    pubchem_ids.append(pubchem_id)
    graph_file_names.append(graph_file_name)

graph_df_kegg = pd.DataFrame({'ID': pubchem_ids, 
                              'SMILES': smile_strings, 
                              'graph_file': graph_file_names,
                              })

logging.info('Saving graph file...')
graph_df_kegg.to_csv('graph_file_df.csv')
logging.info('done.')

logging.info('***')
logging.info('Putting graph files into knowledge graph...')
count = 0
for i in range(graph_df_kegg.shape[0]):
  filename = graph_df_kegg.iloc[i].ID
  try:
    compound_graph.nodes[filename]['graph_file'] = graph_df_kegg.iloc[i].graph_file
  except:
    count += 1
    continue

for mol in compound_graph.nodes:
  mol_graph = nx.read_gpickle('molecules/' + compound_graph.nodes[mol]['graph_file'])
  mol_graph = pyg.utils.convert.from_networkx(mol_graph)
  mol_graph.x = [x.type(torch.LongTensor) for x in mol_graph.el_one_hot]
  compound_graph.nodes[mol]['graph'] = mol_graph

logging.info('done')
logging.info('***')
logging.info('Converting graph to PyG')

g = pyg.utils.convert.from_networkx(compound_graph)
#print(g)

for i in range(len(g.graph)):
  g.graph[i].x = torch.stack(g.graph[i].x).type(torch.FloatTensor)

#####################################################################################

# LEARNING
#####################################################################################
logging.info('Running model...')

gat = GATKGCN(51, 32, 40, 8)
gat_model = GAE(gat)
sage = SAGEKGCN(51, 32, 2)
sage_model = GAE(sage)
gcn = GCNKGCN(51, 32)
gcn_model = GAE(gcn)


graph = train_test_split_edges(g, val_ratio=0.20, test_ratio=0.20)

def train_model(model, epochs, lr, name):
  logging.info('*************** {} ************'.format(name))
  logging.info(graph)
  tr_losses = []
  val_losses = []
  val_aucs = []
  val_aps = []

  opt = torch.optim.Adam(model.parameters(), lr=lr)
  model.train()

  for epoch in range(1, epochs+1):
    opt.zero_grad()
    z = model.encode(graph.graph, graph.train_pos_edge_index)
    loss = model.recon_loss(z, graph.train_pos_edge_index)
    loss.backward()
    opt.step()

    model.eval()
    Z = model.encode(graph.graph, graph.train_pos_edge_index)
    model.eval()
    val_loss = model.recon_loss(Z, graph.val_pos_edge_index)
    auc, ap = model.test(Z, graph.val_pos_edge_index, graph.val_neg_edge_index)
    logging.info('EPOCH: {:03d} TRAIN LOSS: {:.4f} VAL LOSS: {:.4f} VAL AUC: {:.4f} VAL AP: {:.4f}'.format(epoch, 
                                                                                                   loss.item(), 
                                                                                                   val_loss, 
                                                                                                   auc, 
                                                                                                   ap,
                                                                                                   ))


    tr_losses.append(loss.item())
    val_losses.append(val_loss.item())
    val_aucs.append(auc)
    val_aps.append(ap)

  # test
  Z = model.encode(graph.graph, graph.train_pos_edge_index)
  test_auc, test_ap = model.test(Z, graph.test_pos_edge_index, graph.test_neg_edge_index)
  
  logging.info('TEST AUC: {:.4f}        TEST AP: {:.4f}'.format(test_auc, test_ap))

  return tr_losses, val_losses, val_aucs, val_aps, test_auc, test_ap

EPOCHS = 20
LR = 1e-3

gat_tr_loss, gat_val_loss, gat_val_aucs, gat_val_aps, gat_test_auc, gat_test_ap = train_model(gat_model, epochs=EPOCHS, lr=LR, name='GAT')
sage_tr_loss, sage_val_loss, sage_val_aucs, sage_val_aps, sage_test_auc, sage_test_ap = train_model(sage_model, epochs=EPOCHS, lr=LR, name='SAGE')
gcn_tr_loss, gcn_val_loss, gcn_val_aucs, gcn_val_aps, gcn_test_auc, gcn_test_ap = train_model(gcn_model, epochs=EPOCHS, lr=1e-3, name='GCN')

def get_mol_embeddings(model):
  mol_gnn = model.encoder.mol_gnn
  embeds = []
  cids = []
  for m in range(len(graph.graph)):
    mol = graph.graph[m]
    cid = graph.cid[m]
    embeding = mol_gnn(mol.x, mol.edge_index)
    embeds.append(embeding)
    cids.append(cid)
  return torch.stack(embeds, dim=0).detach().numpy(), torch.stack(cids, dim=0).numpy()

plt.figure()
plt.title('AUC')
plt.plot(gat_val_aucs, label='GAT')
plt.plot(sage_val_aucs, label='SAGE')
plt.plot(gcn_val_aucs, label='GCN')
plt.xlabel('epoch')
plt.ylabel('AUC')
plt.legend()
plt.savefig('auc_reaction_graph_one-hot-mol-32-{}'.format(READOUT))

plt.figure()
plt.title('Average Precision')
plt.plot(gat_val_aps, label='GAT')
plt.plot(sage_val_aps, label='SAGE')
plt.plot(gcn_val_aps, label='GCN')
plt.xlabel('epoch')
plt.ylabel('precision')
plt.legend()
plt.savefig('ap_reaction_graph_one-hot-mol-32-{}'.format(READOUT))


plt.figure()
plt.title('Training Loss')
plt.plot(gat_tr_loss, label='GAT')
plt.plot(sage_tr_loss, label='SAGE')
plt.plot(gcn_tr_loss, label='GCN')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('train_losses_reaction_graph_one-hot-mol-32-{}'.format(READOUT))

plt.figure()
plt.title('Validation Loss')
plt.plot(gat_val_loss, label='GAT')
plt.plot(sage_val_loss, label='SAGE')
plt.plot(gcn_val_loss, label='GCN')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('val_losses_reaction_graph_one-hot-mol-32-{}'.format(READOUT))

"""
logging.info(BREAK)
logging.info('Getting molecule embeddings')
embeds, cids = get_mol_embeddings(gcn_model)

cids_to_embeds = {}
for i in range(len(cids)):
  cids_to_embeds[cids[i]] = embeds[i]

#print(get_mol_embeddings(sage_model))
#########################################################################################
logging.info(BREAK)
logging.info(BREAK)
logging.info(BREAK)

logging.info(BREAK + ' KO PREDICTION ' + BREAK)
logging.info(BREAK)
logging.info('Reading in compound-ko relationships...')
cck_df = pd.read_csv('comp_comp_ko')
ko_ids = {}
for i, j in enumerate(set(cck_df.KO)):
  ko_ids[j] = i

c1_embeds = []
c2_embeds = []
ko = []
for i in tqdm.tqdm(range(cck_df.shape[0]), desc='Getting embeddings...'):
  sample = cck_df.iloc[i]
  c1 = int(sample.C1.split(':')[1])
  c2 = int(sample.C2.split(':')[1])
  try:
    cids_to_embeds[c1]
  except:
    continue
  try:
    cids_to_embeds[c2]
  except:
    continue
  try:
    ko_ids[sample.KO]
  except:
    continue

  c1_embeds.append(cids_to_embeds[c1])
  c2_embeds.append(cids_to_embeds[c2])
  ko.append(ko_ids[sample.KO])


  
#logging.info('KOs: {}'.format(len(ko)))
#logging.info('C1s: {}, C2s: {}'.format(len(c1_embeds), len(c2_embeds)))

assert len(ko) == len(c1_embeds)
assert len(c1_embeds) == len(c2_embeds)
assert len(c2_embeds) == len(ko)


class MolData(Dataset):
  def __init__(self, labels, compound1, compound2, label_ids):
    self.labels = labels
    self.c1 = compound1
    self.c2 = compound2
    self.label_ids = label_ids

  def nu_labels(self):
    return len(self.label_ids)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.c1[idx], self.c2[idx], self.labels[idx]

class KOModel(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super(KOModel, self).__init__()
    self.linear1 = torch.nn.Linear(input_dim, 2*input_dim)
    self.linear2 = torch.nn.Linear(2*input_dim, 4*input_dim)
    self.linear3 = torch.nn.Linear(4*input_dim, 8*input_dim)
    self.linear4 = torch.nn.Linear(8*input_dim, output_dim)
  def forward(self, x):
    x = self.linear1(x).relu()
    x = self.linear2(x).relu()
    x = self.linear3(x).relu()
    x = self.linear4(x)
    return x

train_data = MolData(ko, c1_embeds, c2_embeds, ko_ids)
nu_labels = train_data.nu_labels()
print('NUMER OF LABELS {}'.format(nu_labels))
#dataset = torch.utils.data.Subset(train_dataloader, range(0, len(train_dataloader)))
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

model = KOModel(32*2, nu_labels)
opt = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()
epoch_losses = []

# TODO
# accuracy
# validation
for epoch in range(10):
  epoch_loss = 0
  count = 0
  for batch in tqdm.tqdm(train_dataloader):
    opt.zero_grad()
    c1, c2, ko = batch
    out = model(torch.cat((c1, c2), dim=1))
    loss = loss_fn(out, ko)
    loss.backward()
    opt.step()
    epoch_loss += loss.item()
    count += 1
  logging.info('EPOCH: {:03d} TRAIN LOSS: {:.4f} '.format(epoch, epoch_loss / count))
  epoch_losses.append(epoch_loss/len(train_data))

plt.figure()
plt.title('KO Prediction Training Loss')
plt.plot(epoch_losses)
#plt.plot(gat_tr_loss, label='GAT')
#plt.plot(sage_tr_loss, label='SAGE')
#plt.plot(gcn_tr_loss, label='GCN')
#plt.title('One-hot molecules - 32 dimensions')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('train_losses_ko_pred')

# This will print the molecule embddings and the IDs
#mol_embeddings = get_mol_embeddings()
#print(mol_embeddings.detach().numpy())
#print(mol_embeddings.size())
"""
