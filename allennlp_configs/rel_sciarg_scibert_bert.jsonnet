// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).


{
  local data_path = "./dataset_scripts/sciarg.py",
  local transformer_model = "allenai/scibert_scivocab_uncased",
  local transformer_max_length = 512,
  //local transformer_max_length = 128,
  local transformer_hidden_size = 768,

  local seq2seq_transformer_hidden_size = 256,

  //// do not set the following two parameters to get the values from data
  //// note: the run will fail, because seq2vec_encoder requires the correct input_size
  local tag_embedding_size = 13,
  local type_embedding_size = 3,
  "dataset_reader": {
    "type": "brat",
    "instance_type": "R",
    "add_reverted_relations": true,
    "add_negative_relations_portion": 1,
    "entity_coding_scheme": "BIOUL",
    "dataset_splits": {"train": ":20", "dev": "20:30"},
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
      //"token_characters": {
      //  "type": "characters",
      //  "min_padding_length": 3
      //},
    },
  },
  "train_data_path": data_path + '@train',
  "validation_data_path": data_path + '@dev',
  //"test_data_path": data_path + '@test',
  "model": {
    "type": "basic_typed_classifier",
    //"dropout": 0.5,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            "max_length": transformer_max_length,
            //"train_parameters": true
            "train_parameters": false
        },
      },
    },
    "tag_embedding_size": tag_embedding_size,
    "type_embedding_size": type_embedding_size,
    "seq2seq_encoder": {
        "type": "compose",
        "encoders": [
            // rescale to match trasnformer input size
            {
                "type": "feedforward",
                "feedforward": {
                    "input_dim": transformer_hidden_size + type_embedding_size + tag_embedding_size,
                    "num_layers": 1,
                    "hidden_dims": seq2seq_transformer_hidden_size,
                    "activations": "relu",
                }
            },
            // apply transformer
            {
                "type": "pytorch_transformer",
                "input_dim": seq2seq_transformer_hidden_size,
                "feedforward_hidden_dim": 1024, // scibert intermediate_size: 3072
                "num_layers": 3,
                "num_attention_heads": 8,
                //"use_positional_encoding": none, // default: none
                //"dropout_prob": 0.0, // default: 0.1
            },
        ],
     },
    "seq2vec_encoder": {
        "type": "cls_pooler",
        "embedding_dim": seq2seq_transformer_hidden_size
    }
  },
  "data_loader": {
    //"batch_size": 64
    "batch_size": 4,
    "shuffle": true,
    //"num_workers": 4,
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
    "validation_metric": "+micro-f1/fscore",
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25
  }
}