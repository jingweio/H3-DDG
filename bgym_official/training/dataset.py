
import copy
import numpy as np
import torch
from torch_geometric.data import HeteroData,Dataset
from protein_mpnn_utils import parse_PDB,tied_featurize

class StructureDataset(Dataset):
    def __init__(self, df, idxs, structure_path,batch_size,esm_alphabet, evaluation=True, predict=False, seed_bias=0):
        super().__init__()
        if not predict:
            self.stage = 'train'
        else:
            self.stage = 'test'
        self.df = df
        self.batch_size = batch_size
        self.evaluation = evaluation
        self.predict = predict
        self.idxs = idxs
        self.seed_bias = seed_bias
        if not evaluation:
            self.n = 0
            self.batch = []
            for idx in idxs:
                self.n += df[idx].shape[0]
                poi = df[idx]['POI'].values[0]
                pdb_dict_list = parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
                if 0:
                    designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
                else:
                    designed_chain_list = all_chain_list
                fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
                chain_id_dict = {}
                chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
                self.batch.append(tied_featurize(pdb_dict_list, 'cpu', chain_id_dict, None, None, None, None, None, ca_only=False))
            print('train size:',self.n)
        else:
            pdb_dict_list = []
            chain_id_dict = {}
            self.poi_dic = {}
            for i,poi in enumerate(df['POI'].unique()):
                self.poi_dic[poi] = i
                pdb_dict_list += parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                all_chain_list = sorted([item[-1:] for item in list(pdb_dict_list[-1]) if item[:9]=='seq_chain']) #['A','B', 'C',...]
                designed_chain_list = all_chain_list
                fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
                chain_id_dict[pdb_dict_list[-1]['name']]= (designed_chain_list, fixed_chain_list)
            self.batch = tied_featurize(pdb_dict_list, 'cpu', chain_id_dict, None, None, None, None, None, ca_only=False)
        
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(alphabet, range(21))) 
    def __len__(self):
        if self.evaluation:
            return (len(self.idxs))
        else:
            return (min(self.batch_size*256,self.n))

    def __getitem__(self, index):
        if self.evaluation:
            idx = self.idxs[index]
            seq = self.df.loc[idx,'mutated_sequence']
            mseq = list(seq)
            mutant = self.df.loc[idx,'mutant']
            poi = self.df.loc[idx,'POI']
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = self.batch
            X = X[[self.poi_dic[poi]]]
            S = S[[self.poi_dic[poi]]]
            mask = mask[[self.poi_dic[poi]]]
            chain_M = chain_M[[self.poi_dic[poi]]]
            residue_idx = residue_idx[[self.poi_dic[poi]]]
            chain_encoding_all = chain_encoding_all[[self.poi_dic[poi]]]
            reg_label = -self.df.loc[idx,'ddg'] if 'DMS_score' not in self.df.columns else self.df.loc[idx,'DMS_score']
            # bin_label = self.df.loc[idx,'DMS_score_bin']
        else:
            # todo: 可以调整每个DMS数据的权重训练
            seed = index // self.batch_size
            np.random.seed(seed+self.seed_bias*1000000)
            select_idx = np.random.randint(0,len(self.df))
            df = self.df[select_idx]
            np.random.seed(index+self.seed_bias*1000000)
            idx = np.random.randint(0,len(df))
            seq = df.loc[idx,'mutated_sequence']
            mseq = list(seq)
            mutant = df.loc[idx,'mutant']
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = self.batch[select_idx]
            reg_label = df.loc[idx,'DMS_score']
            # bin_label = df.loc[idx,'DMS_score_bin']


        mpos = []
        if mutant != '':
            for m in mutant.split(':'):
                pos = int(m[1:-1])
                mpos.append(pos)
                if not self.evaluation:
                    mseq[pos-1] = 'X'
        mseq = ''.join(mseq)

        # chain_M_pos = copy.deepcopy(chain_M)
        # n = 0
        # for i,a in enumerate(chain_M_pos[0]):
        #     if a == 1:
        #         n += 1
        #     if n not in mpos:
        #         chain_M_pos[0][i] = 0
        graph = HeteroData()
        
        graph['protein'].X = X
        graph['protein'].wt_S = S 
        graph['protein'].S = copy.deepcopy(S)
        S_input = torch.tensor([self.alphabet_dict[AA] for AA in mseq])

        graph['protein'].S[0,:len(seq)] = S_input
        graph['protein'].mask = mask
        graph['protein'].chain_M = chain_M
        # graph['protein'].chain_M_pos = chain_M_pos
        graph['protein'].residue_idx = residue_idx
        graph['protein'].chain_encoding_all = chain_encoding_all
        graph['batch'].x = torch.ones(len(seq),1)
        target_S = copy.deepcopy(S)
        target_S[0,:len(seq)] = torch.tensor([self.alphabet_dict[AA] for AA in seq])
        graph.token_idx = torch.cat([S,target_S],dim=0).long()[None,...,None]

        if not self.predict:
            graph.reg_labels = torch.tensor([[reg_label]]).float()
            # graph.bin_labels = torch.tensor([[bin_label]]).float()
        
        return graph

class SequenceDataset(Dataset):
    def __init__(self, df, idxs, structure_path, batch_size, esm_alphabet, evaluation=True, predict=False, seed_bias=0):
        super().__init__()
        if not predict:
            self.stage = 'train'
        else:
            self.stage = 'test'
        self.df = df
        self.batch_size = batch_size
        self.esm_alphabet = esm_alphabet
        self.esm_alphabet_dic = esm_alphabet.to_dict()
        self.evaluation = evaluation
        self.predict = predict
        self.idxs = idxs
        self.seed_bias = seed_bias
        if not evaluation:
            self.n = 0
            for idx in idxs:
                self.n += df[idx].shape[0]
            
        '''
        if not evaluation:
            self.n = 0
            self.batch = []
            for idx in idxs:
                self.n += df[idx].shape[0]
                poi = df[idx]['POI'].values[0]
                pdb_dict_list = parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
                if 0:
                    designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
                else:
                    designed_chain_list = all_chain_list
                fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
                chain_id_dict = {}
                chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)
                self.batch.append(tied_featurize(pdb_dict_list, 'cpu', chain_id_dict, None, None, None, None, None, ca_only=False))
            print('train size:',self.n)
        else:
            pdb_dict_list = []
            chain_id_dict = {}
            self.poi_dic = {}
            for i,poi in enumerate(df['POI'].unique()):
                self.poi_dic[poi] = i
                pdb_dict_list += parse_PDB(f'{structure_path}/{poi}.pdb', ca_only=False)
                all_chain_list = [item[-1:] for item in list(pdb_dict_list[-1]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
                if 0:
                    designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
                else:
                    designed_chain_list = all_chain_list
                fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
                chain_id_dict[pdb_dict_list[-1]['name']]= (designed_chain_list, fixed_chain_list)
            self.batch = tied_featurize(pdb_dict_list, 'cpu', chain_id_dict, None, None, None, None, None, ca_only=False)
        '''
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        self.alphabet_dict = dict(zip(alphabet, range(21))) 
    def __len__(self):
        if self.evaluation:
            return (len(self.idxs))
        else:
            return (min(self.batch_size*256,self.n))

    def __getitem__(self, index):
        if self.evaluation:
            idx = self.idxs[index]
            seq = self.df.loc[idx,'mutated_sequence']
            mseq = list(seq)
            mutant = self.df.loc[idx,'mutant']
            reg_label = -self.df.loc[idx,'ddg'] if 'DMS_score' not in self.df.columns else self.df.loc[idx,'DMS_score']
            # bin_label = self.df.loc[idx,'DMS_score_bin']
        else:
            # todo: 可以调整每个DMS数据的权重训练
            seed = index // self.batch_size
            np.random.seed(seed+self.seed_bias*1000000)
            select_idx = np.random.randint(0,len(self.df))
            df = self.df[select_idx]
            np.random.seed(index+self.seed_bias*1000000)
            idx = np.random.randint(0,len(df))
            seq = df.loc[idx,'mutated_sequence']
            mseq = list(seq)
            mutant = df.loc[idx,'mutant']
            reg_label = df.loc[idx,'DMS_score']
            # bin_label = df.loc[idx,'DMS_score_bin']


        if mutant != '':
            for m in mutant.split(':'):
                mseq[int(m[1:-1])-1] = m[0]
        base_seq = ''.join(mseq)

        esm_token_idxs = []
        esm_base_token_idxs = []
        esm_mask_seq = ''
        n = 0
        for i,aa in enumerate(seq):
            if aa != base_seq[i] or (i==len(seq)-1 and len(esm_token_idxs)==0):
                n +=1
                esm_mask_seq += '<mask>'
                esm_token_idxs.append(self.esm_alphabet_dic[aa])
                esm_base_token_idxs.append(self.esm_alphabet_dic[base_seq[i]])
            else:
                esm_mask_seq += aa
        graph = HeteroData()
        
        graph.esm_token_idx = torch.tensor([esm_base_token_idxs,esm_token_idxs]).long().T

        # N(bs), L(length)
        esm_x = self.esm_alphabet.get_batch_converter()([('',esm_mask_seq)])
        graph['protein'].x = esm_x[2][0]
        if not self.predict:
            graph.reg_labels = torch.tensor([[reg_label]]).float()
        
        return graph