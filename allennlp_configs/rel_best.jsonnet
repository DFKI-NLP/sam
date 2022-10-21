{
  local data_path = "./dataset_scripts/sciarg.json",
  local transformer_model = "allenai/scibert_scivocab_uncased",
  local seed = 13370,
  local numpy_seed = 1337,
  local torch_seed = 133,
  local transformer_max_length = 512,
  local transformer_hidden_size = 768,

  //// do not set the following two parameters to get the values from data
  //// note: the run will fail, because seq2vec_encoder requires the correct input_size
  local tag_embedding_size = 13,
  local type_embedding_size = 3,
  "dataset_reader": {
    "type": "brat",
    "instance_type": "R",
    "symmetric_relations": ["parts_of_same", "semantically_same", "contradicts"],
    "add_reverted_relations": true,
    "add_negative_relations_portion": 3.0,
    "max_argument_distance": 177,
    "token_window_size":479,
    "entity_coding_scheme": "BIOUL",
    "dataset_splits": {"train": ":20", "dev": "20:30"},
    "relation_type_blacklist": ["semantically_same"],
    // remove everything before "<title>":
    "delete_pattern": "<\\?xml[^>]*>[^<]*<Document xmlns:gate=\"http://www.gate.ac.uk\"[^>]*>[^<]*",
    // remove all (xml-)tags:
    //"delete_pattern": "<[^>]*>",
    "split_pattern": "<H1>",
    //// for debugging call with: -o {dataset_reader:{split_creation:{train:\":1\",dev:\"20:21\"}}}
    //"split_creation": {
    //    "train": ":1",
    //    "dev": "20:21"
    //},
    "tokenizers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": transformer_max_length
      }
    },
  },
  "train_data_path": data_path + '@train',
  "validation_data_path": data_path + '@dev',
  //"test_data_path": data_path + '@test',
  "model": {
    "type": "basic_typed_classifier",
    "dropout": 0.3061,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": transformer_max_length,
            "train_parameters": false
        },
       },
    },
    "tag_embedding_size": tag_embedding_size,
    "type_embedding_size": type_embedding_size,
    "seq2seq_encoder": {
        "type": "lstm",
        //"input_size": 50 + 128,
        "input_size": transformer_hidden_size + type_embedding_size + tag_embedding_size,
        "hidden_size": 430,
        "num_layers": 4,
        "dropout": 0.2566,
        "bidirectional": true
    },
    "seq2vec_encoder": {
        "type": "cnn",
        "embedding_dim": 860,
        "num_filters": 193,
        "ngram_filter_sizes": [3, 5, 7, 10],
    }
  },
  "data_loader": {
    "batch_size": 128,
    "shuffle": true,
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.0005
    },
   // "checkpointer": {
   //     "num_serialized_models_to_keep": 1,
   // },
    "validation_metric": "+micro-f1/fscore",
    "num_epochs": 75,
    "grad_norm": 4.12,
    "patience": 20,
    "callbacks": [
    {
    "type": "wandb",
     'entity': 'sam',
     'project': 'rel_sci_arg_init_testing'
     },
  ],
  }
}
