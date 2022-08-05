def attach_model_arguments(parser):
    ## Required parameters
    parser.add_argument("--train_data_file_s3", default='data/news-discourse-training-data.csv', type=str)
    parser.add_argument("--eval_data_file_s3", default=None, type=str)
    parser.add_argument("--pretrained_files_s3", default='elmo', type=str)
    parser.add_argument("--finetuned_lm_file", default=None, type=str, help="If you fine-tune the LM.")
    parser.add_argument("--processed_data_fname", default=None, type=str, help="Where to upload the results.")
    parser.add_argument("--discriminator_path", "-D", type=str, default=None, help="Discriminator to use.")
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--batch_size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--do_eval', action="store_true")
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--env', type=str, default='local')
    parser.add_argument('--num_dataloader_cpus', type=int, default=10, help='Number of CPUs to use to process data')
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use per node.")
    parser.add_argument('--num_nodes', type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument('--experiment', type=str, default='baseline_non-sequential', help="Which experiment to run.")
    parser.add_argument('--split_type', type=str, default='key')
    parser.add_argument('--train_perc', type=float, default=1.0)
    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--log_all_metrics', action='store_true')
    parser.add_argument('--tb_logdir', default=None, type=str, help="Path for tensorboard logs.")

    ## optimization params
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=float, default=0, help="Num warmup steps.")
    parser.add_argument("--weight_decay", type=float, default=0, help="Whether to have a LR decay schedule or not (not implemented).")
    parser.add_argument("--max_grad_norm", type=float, default=0, help="Clip the gradients at a certain point.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients across batches.")
    parser.add_argument('--adam_beta1', type=float, default=.9)
    parser.add_argument('--adam_beta2', type=float, default=.999)
    parser.add_argument('--adam_epsilon', type=float, default=1e-08)

    ## model params
    #### general model params
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    parser.add_argument('--freeze_encoder_layers', nargs="*", default=[])
    parser.add_argument('--freeze_pooling_layer', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--sentence_embedding_method', type=str, default='cls', help="['cls', 'average']")
    parser.add_argument('--max_length_seq', type=int, default=512, help='How to generate document embeddings.')
    parser.add_argument('--max_num_sentences', type=int, default=100, help='How to generate document embeddings.')
    parser.add_argument('--max_num_word_positions', type=int, default=1024, help="How many positional embeddings for GPT2.")
    parser.add_argument('--label_context_back', type=int, default=0, help='How many labels beforehand to include in prediction.')
    parser.add_argument('--label_context_forward', type=int, default=0, help='How many labels beforehand to include in prediction.')
    parser.add_argument('--num_labels_pred_window', type=int, default=None, help='Whether to do a multi-step prediction problem.')
    parser.add_argument('--label_pos_embs', action='store_true', help="Whether to use positional embeddings with the labels.")

    # contextualizing layer
    parser.add_argument('--context_layer', type=str, default='lstm', help="How to provide context to sentence vectors. {lstm, gpt2-sentence}")
    parser.add_argument('--num_contextual_layers', type=int, default=1)
    parser.add_argument('--bidirectional', action='store_true', help="If LSTM only, whether to be bidirectional or not.")
    parser.add_argument('--num_sent_attn_heads', type=int, help="If Transformer context layer only, how many attention heads to have in each layer.")

    # classifier enhancements
    parser.add_argument('--use_headline', action='store_true')
    parser.add_argument('--use_headline_embs', action='store_true')
    parser.add_argument('--concat_headline', action='store_true')
    parser.add_argument('--use_positional', action='store_true')
    parser.add_argument('--sinusoidal_embeddings', action='store_true')
    parser.add_argument('--max_num_sent_positions', type=int, default=40)
    parser.add_argument('--use_doc_emb', action='store_true')
    parser.add_argument('--doc_embed_arithmetic', action='store_true')
    parser.add_argument('--do_doc_pred', action='store_true', default=False, help='TEMPORARY (for edits project)')
    parser.add_argument('--do_version', action='store_true', default=False, help='TEMPORARY (for edits project)')
    parser.add_argument('--separate_heads', action='store_true', default=False, help='Keep the full head layer separate across tasks.')
    parser.add_argument('--share_label_embeds', action='store_true', default=False, help='Whether to share the label embedding layer across tasks.')
    parser.add_argument('--use_y', action='store_true', default=False, help='Use label for classifier (for ablation)')

    # multitask
    parser.add_argument('--do_multitask', action='store_true', help='TEMPORARY (for edits project)')
    parser.add_argument('--loss_weighting', nargs="*", default=[])

    # scoring
    parser.add_argument('--map_tags', action='store_true', help='whether to output class idxs or names.')
    parser.add_argument('--num_output_tags', type=int, default=None, )
    parser.add_argument('--label_col',  nargs="*", default=[])
    return parser

def formulate_module_args(args):
    top_level_arguments = [
        'job_script_module',
        'package_uri',
        'job_size',
        'branch',
        'git_identity_id',
        'hadoop_identity_id',
        'gen_name',
        'n_gpus'
    ]

    module_args = []
    arg_vars = vars(args)
    for param, value in arg_vars.items():
        if param in top_level_arguments:
            continue

        ## exceptions
        if param == 'notes':
            module_args.extend(['--notes', '_'.join(value.split())])

        ## general rules
        elif isinstance(value, str):
            module_args.extend(["--%s" % param, value])
        elif isinstance(value, bool):
            if value:
                module_args.append('--%s' % param)
        elif isinstance(value, int) or isinstance(value, float):
            module_args.extend(["--%s" % param, str(value)])
        elif isinstance(value, list):
            module_args.extend(["--%s" % param, ' '.join(value)])

    return module_args








#     #### semisupervised arguments
#     parser.add_argument('--num_augmentations', type=int, default=None)
#     parser.add_argument('--perc_unsup_files', type=float, default=None)
#     ## uda parameters
#     parser.add_argument("--unsupervised_data_file", default=None, type=str)
#     parser.add_argument('--use_tsa', action='store_true')
#     parser.add_argument('--tsa_schedule', type=str, default='linear_schedule')
#     parser.add_argument('--uda_softmax_temp', type=float, default=0.85)
#     parser.add_argument('--uda_confidence_thresh', type=float, default=0.45) # 9 .... 1/9 == .18
#     parser.add_argument('--uda_coeff', type=float, default=1)
#     parser.add_argument('--use_class_scaling', action='store_true')
#     parser.add_argument('--class_scale_beta', type=float, default=1)