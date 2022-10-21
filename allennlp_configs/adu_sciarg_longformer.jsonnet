// Configuration for a named entity recognization model based on:
//   Peters, Matthew E. et al. “Deep contextualized word representations.” NAACL-HLT (2018).


{
  local transformer_model = "allenai/longformer-base-4096",
  local transformer_max_length = 4096,
  //local transformer_max_length = 128,
  local transformer_hidden_size = 768,
  local data_path = "./dataset_scripts/sciarg.py",
  "dataset_reader": {
    "type": "brat",
    "instance_type": "T",
    "entity_coding_scheme": "BIOUL",
    "dataset_splits": {"train": ":20", "dev": "20:30"},
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
    "type": "crf_tagger",
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
        //"token_characters": {
        //    "type": "character_encoding",
        //    "embedding": {
        //        "embedding_dim": 16
        //    },
        //    "encoder": {
        //        "type": "cnn",
        //        "embedding_dim": 16,
        //        "num_filters": 128,
        //        "ngram_filter_sizes": [3],
        //        "conv_layer_activation": "relu"
        //    }
        //  }
       },
    },
    "encoder": {
        "type": "lstm",
        //"input_size": 50 + 128,
        "input_size": transformer_hidden_size,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.5,
        "bidirectional": true
    },
  },
  "data_loader": {
    //"batch_size": 64
    "batch_size": 2
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 1,
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 25
  }
}