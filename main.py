from data_loader.data_loader import Preprocessor
from models.cnn_model import SimpleVGG
from models.vit_model import SimpleViT
from training.trainer import Trainer
from testing.tester import Tester
from utils.utils import read_yaml
import argparse
import logging


def main(config_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = read_yaml(config_path)

    # Data loading and preprocessing
    preprocessor = Preprocessor(image_size=(config['data']['image_size'], config['data']['image_size']), batch_size=config['data']['batch_size'])
    train_loader, test_loader = preprocessor.get_data_loaders()

    # CNN Model
    logging.info("Training CNN Model...")
    cnn_model = SimpleVGG()
    cnn_trainer = Trainer(cnn_model, train_loader, epochs=config['cnn_model']['epochs'], lr=config['cnn_model']['lr'])
    cnn_trainer.train()
    cnn_tester = Tester(cnn_model, test_loader)
    y_true_cnn, y_pred_cnn = cnn_tester.test()
    cnn_tester.analyze(y_true_cnn, y_pred_cnn)

    # Vision Transformer Model
    logging.info("Training Vision Transformer Model...")
    vit_model = SimpleViT(
        embed_dim=config['vit_model']['embed_dim'],
        depth=config['vit_model']['depth'],
        n_heads=config['vit_model']['n_heads'],
        mlp_ratio=config['vit_model']['mlp_ratio'],
        qkv_bias=config['vit_model']['qkv_bias'],
        p=config['vit_model']['p'],
        attn_p=config['vit_model']['attn_p']
    )
    vit_trainer = Trainer(vit_model, train_loader, epochs=config['vit_model']['epochs'], lr=config['vit_model']['lr'])
    vit_trainer.train()
    vit_tester = Tester(vit_model, test_loader)
    y_true_vit, y_pred_vit = vit_tester.test()
    vit_tester.analyze(y_true_vit, y_pred_vit)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default_conf.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)
