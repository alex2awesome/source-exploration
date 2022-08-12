def attach_model_arguments(parser):
    # setup params
    parser.add_argument('--env', type=str, default='local', help='Whether to download from BB or not.')
    parser.add_argument('--train_data_file', type=str, help='What dataset to use.')
    parser.add_argument('--eval_data_file', type=str, help='What dataset to use.')
    parser.add_argument('--pretrained_model_path', type=str, help='What lm to use.')
    parser.add_argument('--pretrained_lm_model_path', type=str, default=None, help="if there's a specific LM model checkpoint to use.")
    parser.add_argument('--experiment', type=str, default=None, help='Which model/dataset classes to use.')
    parser.add_argument('--notes', type=str, help='Notes about the experiment.')
    parser.add_argument('--local_rank', type=int, required=False, default=-1, help='Local rank when doing multi-process training, set to -1 if not')
    parser.add_argument('--model_type', type=str, default='roberta')
    parser.add_argument('--sentence_contextualizer_model_type', type=str, default='roberta')
    parser.add_argument('--spacy_path', type=str, default=None)
    parser.add_argument('--log_all_metrics', action='store_true')
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--downsample_negative_data', default=1.0, type=float, help="rate at which to downsample negative datapoints.")

    # hardware params
    parser.add_argument('--num_dataloader_cpus', type=int, default=10, help='Number of CPUs to use to process data')
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use per node.")
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of nodes to use.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed in Pytorch Lightning.")
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Transformer-native gradient checkpointing.')

    # dataset params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--max_length_seq', type=int, default=200)
    parser.add_argument('--max_num_sentences', type=int, default=100)
    parser.add_argument('--num_documents', type=int, default=None)
    parser.add_argument('--num_token_types', type=int, default=3)
    parser.add_argument('--max_num_word_positions', type=int, default=2048)

    # optimization params
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument("--num_warmup_steps", type=float, default=0, help="Num warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Whether to have a LR decay schedule or not (not implemented).")
    parser.add_argument("--max_grad_norm", type=float, default=0, help="Clip the gradients at a certain point.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients across batches.")
    parser.add_argument('--adam_beta1', type=float, default=.9)
    parser.add_argument('--adam_beta2', type=float, default=.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)
    parser.add_argument('--hidden_dim', type=int, default=512)

    # model parameters
    parser.add_argument('--sentence_embedding_method', default='multiheaded-attention')
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--freeze_encoder_layers', nargs="*", default=[])
    parser.add_argument('--freeze_pooling_layer', action='store_true')

    # classifier enhancements
    parser.add_argument('--use_headline', action='store_true')
    parser.add_argument('--use_headline_embs', action='store_true')
    parser.add_argument('--concat_headline', action='store_true')
    parser.add_argument('--use_positional', action='store_true')
    parser.add_argument('--sinusoidal_embeddings', action='store_true')
    parser.add_argument('--max_position_embeddings', type=int, default=40)
    parser.add_argument('--use_doc_emb', action='store_true')
    parser.add_argument('--doc_embed_arithmetic', action='store_true')

    return parser