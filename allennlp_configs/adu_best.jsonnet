// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).

{
  local transformer_model = "allenai/scibert_scivocab_uncased",
  local transformer_max_length = 512,
  local transformer_hidden_size = 768,
  local seed = 13370,
  local numpy_seed = 1337,
  local torch_seed = 133,
  local data_path = "./dataset_scripts/sciarg.py",
  "dataset_reader": {
    "type": "brat",
    "instance_type": "T",
    "entity_coding_scheme": "BIOUL",
    "dataset_splits": {"train": ":20", "dev": "20:30"},
    // remove everything before "<title>":
    "delete_pattern": "<\\?xml[^>]*>[^<]*<Document xmlns:gate=\"http://www.gate.ac.uk\"[^>]*>[^<]*",
    // remove all (xml-)tags:
    //"delete_pattern": "<[^>]*>",
    "split_pattern": "<H1>",
    //// for debugging call with: -o {dataset_reader:{split_creation:{train:\":5\",dev:\"20:23\"}}}
    // "split_creation": {
    //   "train": ":5",
    //   "dev": "20:23"
    //  },
    "tokenizers":{
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
      },
    },
  },
  "train_data_path": data_path + '@train',
  "validation_data_path": data_path + '@dev',
  //"test_data_path": data_path + '@test',
  "model": {
    "type": "crf_tagger_with_f1",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
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
    "encoder": {
        "type": "lstm",
        //"input_size": 50 + 128,
        "input_size": transformer_hidden_size,
        "hidden_size": 300,
        "num_layers": 2,
        "dropout": 0.4394,
        "bidirectional": true
    },
  },
  "data_loader": {
    "batch_size": 8
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.005
    },
    //"checkpointer": {
     //   "num_serialized_models_to_keep": 1,
   // },
    "validation_metric": "+span/overall/f1",
    "num_epochs": 75,
    "grad_norm": 7.0,
    "patience": 20,
    "callbacks": [
        {
        "type": "custom_wandb",
         'entity': std.extVar('WANDB_ENTITY'),
         'project': std.extVar('WANDB_PROJECT'),
         'files_to_save': ["config.json", "out.log","metrics.json"]
         },
  ],
  }
}
