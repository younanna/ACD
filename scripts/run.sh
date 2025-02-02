gpu_n=$1
config_name=$2

CUDA_VISIBLE_DEVICES=$gpu_n python -m pipeline.generate \
    --config-path=../configs \
    --config-name=$config_name.yaml \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    hydra/hydra_logging=disabled \

python -m pipeline.evaluate \
    --config-path=../configs \
    --config-name=$config_name.yaml \
    hydra.run.dir=. \
    hydra.output_subdir=null \
    hydra/hydra_logging=disabled \
