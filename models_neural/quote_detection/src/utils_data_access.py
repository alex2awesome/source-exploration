import logging
import os

access_key = 'VN2M29BH4PCAJT9ABZKN'
secret_access_key = 'O9bcBpaprrXr6Q3dorn0XYI4Kp8go6oBDBYFYqeD'
endpoint = 'http://s3.dev.obdc.bcs.bloomberg.com'
os.environ['AWS_ACCESS_KEY_ID'] = access_key
os.environ['AWS_SECRET_ACCESS_KEY'] = secret_access_key
os.environ['AWS_ENDPOINT'] = endpoint


def is_local(args):
    if hasattr(args, 'local'):
        return args.local

    if hasattr(args, 'env'):
        return args.env == 'local'


def get_lm_path(args):
    if hasattr(args, 'pretrained_model_path'):
        return args.pretrained_model_path
    if hasattr(args, 'pretrained_lm_model_path'):
        return args.pretrained_lm_model_path


def get_discrim_path(args):
    if hasattr(args, 'discriminator_path'):
        return args.discriminator_path
    if hasattr(args, 'pretrained_discriminator_path'):
        return args.pretrained_discriminator_path
    if hasattr(args, 'pretrained_discrim_path'):
        return args.pretrained_discrim_path


def get_pretrained_editor_lm_path(args):
    if hasattr(args, 'pretrained_editor_lm_path'):
        return args.pretrained_editor_lm_path

    if hasattr(args, 'editor_pretrained_editor_lm_path'):
        return args.editor_pretrained_editor_lm_path


def get_finetuned_editor_lm_path(args):
    if hasattr(args, 'finetuned_editor_lm_path'):
        return args.finetuned_editor_lm_path

    if hasattr(args, 'editor_finetuned_editor_lm_path'):
        return args.editor_finetuned_editor_lm_path


def download_all_necessary_files(args):
    # Download model files
    if get_lm_path(args) is not None:
        if not is_local(args):
            logging.info('Downloading pretrained LM: %s...' % get_lm_path(args))
            download_model_files_bb(get_lm_path(args))

    if get_pretrained_editor_lm_path(args):
        if not is_local(args):
            logging.info('Downloading pretrained LM for editor: %s...' % get_pretrained_editor_lm_path(args))
            download_model_files_bb(get_pretrained_editor_lm_path(args))

    if get_discrim_path(args):
        if not is_local(args):
            logging.info('Downloading pretrained discriminator: %s...' % get_discrim_path(args))
            download_file_to_filepath(get_discrim_path(args))

    # Download Spacy
    if hasattr(args, 'spacy_model_file') and args.spacy_model_file is not None:
        import spacy
        if not is_local(args):
            logging.info('Downloading spacy model file: %s...' % args.spacy_model_file)
            download_model_files_bb(args.spacy_model_file, local_path='en_core', use_zip=False, use_pretrained_dir=False)
            args.spacy_model = spacy.load('en_core')
            args.spacy_model_file = 'en_core'
        else:
            args.spacy_model = spacy.load(args.spacy_model_file)

    # Download data file
    if hasattr(args, 'real_data_file') and args.real_data_file is not None:
        # download data file
        if not is_local(args):
            logging.info('Downloading data file')
            download_file_to_filepath(remote_file_name=args.real_data_file)

    if hasattr(args, 'pretrained_lm_model_path') and args.pretrained_lm_model_path is not None:
        if not is_local(args):
            download_file_to_filepath(args.pretrained_lm_model_path)

    if hasattr(args, 'config_path') and args.config_path is not None:
        if not is_local(args):
            download_file_to_filepath(remote_file_name=args.config_path)

    if hasattr(args, 'auxiliary_data_file') and args.auxiliary_data_file is not None:
        if not is_local(args):
            download_file_to_filepath(remote_file_name=args.auxiliary_data_file)

    if get_finetuned_editor_lm_path(args):
        if not is_local(args):
            logging.info('Downloading finetuned Editor: %s...' % get_finetuned_editor_lm_path(args))
            download_file_to_filepath(remote_file_name=get_finetuned_editor_lm_path(args))

    if hasattr(args, 'edit_discriminator_path') and args.edit_discriminator_path is not None:
        if not is_local(args):
            download_file_to_filepath(remote_file_name=args.edit_discriminator_path)

    if hasattr(args, 'edit_discrim_config_path') and args.edit_discrim_config_path is not None:
        if not is_local(args):
            download_file_to_filepath(remote_file_name=args.edit_discrim_config_path)

    return args


def get_fs():
    import s3fs
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://s3.dev.obdc.bcs.bloomberg.com'})
    return fs


def download_model_files_bb(remote_model, use_pretrained_dir=True, local_path=None, use_zip=True):
    """
    Download pretrained model files from bb S3.

    params:
    * remote_model_path: s3 directory, (or filename, if `use_pretrained_dir`=True)
    * use_pretrained_dir: whether to look in aspangher/transformer-pretrained-models or not
    * local_path: where to download the model files. If none, default to the basename of `remote_model_path`
    * use_zip: whether to unzip the model directory or not.
        If `use_zip` is False and model_path ends in a `/`, then `fs.get()` is called recursively, otherwise, not.
    """
    if local_path is None:
        local_path = remote_model

    fs = get_fs()

    # format model name/path
    model_file_name = '%s.zip' % remote_model if use_zip else remote_model
    model_path_name = 'aspangher/transformer-pretrained-models/%s' % model_file_name if use_pretrained_dir else model_file_name

    print('downloading %s -> %s...' % (model_path_name, local_path))
    # download and optionally unzip
    if use_zip:
        fs.get(model_path_name, '%s.zip' % local_path)
        import zipfile
        with zipfile.ZipFile('%s.zip' % local_path, 'r') as zip_ref:
            zip_ref.extractall()
    else:
        recursive = False
        if remote_model.strip()[-1] == '/':
            recursive = True
        fs.get(model_path_name, '%s' % local_path, recursive=recursive)

    print('downloaded models, at: %s' % local_path)


def download_file_to_filepath(remote_file_name, local_path=None):
    if local_path is None:
        local_path = remote_file_name
    #
    fs = get_fs()
    local_file_dir = os.path.dirname(local_path)
    if local_file_dir != '':
        os.makedirs(local_file_dir, exist_ok=True)
    if 's3://' in remote_file_name:
        remote_file_name = remote_file_name.replace('s3://', '')
    if 'aspangher' not in remote_file_name or 'controlled-sequence-gen' not in remote_file_name:
        remote_file_name = os.path.join('aspangher', 'controlled-sequence-gen', remote_file_name)
    fs.get(remote_file_name, local_path)
    print('Downloading remote filename %s -> %s' % (remote_file_name, local_path))
    return local_path

def upload_file_to_filepath(local_name, remote_file_name=None):
    if remote_file_name is None:
        remote_file_name = local_name
    #
    fs = get_fs()
    if 's3://' in remote_file_name:
        remote_file_name = remote_file_name.replace('s3://', '')
    if 'aspangher' not in remote_file_name or 'controlled-sequence-gen' not in remote_file_name:
        remote_file_name = os.path.join('aspangher', 'controlled-sequence-gen', remote_file_name)
    fs.upload(local_name, remote_file_name)
    print('Uploading file %s -> %s' % (local_name, remote_file_name))
    return local_name