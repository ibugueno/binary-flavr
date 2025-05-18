
# Binary-FLAVR: Binarized Flow-Agnostic Video Representations for Fast Frame Interpolation

> Basado en [FLAVR (WACV 2023 Best Paper Finalist)](https://tarun005.github.io/FLAVR/), esta versión implementa **binarización completa de convoluciones** usando `BinConv2d` y `BinConv3d`.

[[Original Paper](https://arxiv.org/pdf/2012.08512.pdf)] | [[Project Video](https://youtu.be/HFOY7CGpJRM)]

---

## ✨ ¿Qué es Binary-FLAVR?

**Binary-FLAVR** es una variante del modelo FLAVR original que reemplaza las convoluciones estándar con versiones completamente binarizadas (inspiradas en XNOR-Net). El objetivo es reducir el consumo de memoria y acelerar la inferencia, manteniendo un desempeño competitivo para tareas de interpolación de video.

Se binarizan:
- La arquitectura encoder-decoder (spatio-temporal 3D convolutions).
- La fase de entrenamiento, usando `BinOp` para el manejo de pesos binarios.

---

## 📦 Estructura del Proyecto

```
binary-flavr/
├── main_bin.py               # Entrenamiento binarizado
├── model/
│   ├── binconv.py            # Módulos de convolución binarizada
│   ├── binFLAVR_arch.py      # Versión binarizada del UNet_3D_3D
│   ├── binresnet_3D.py       # Versión binarizada del encoder 3D
│   └── ...
├── loss.py                   # Función de pérdida (no binarizada)
├── util.py                   # BinOp para manejar pesos binarios
├── test.py                   # Evaluación estándar
├── interpolate.py            # Interpolación en videos reales
```

---

## 🧩 Dependencias

Probado con:

- Python: 3.10.16
- PyTorch: 2.7.0+cu118
- TorchVision: 0.22.0+cu118
- NumPy: 1.24.4
- OpenCV-Python: 4.11.0.86
- TQDM: 4.67.1


---

## 🏋️‍♂️ Entrenamiento Binarizado

### Vimeo-90K (2x interpolación):

```bash
wget http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
```

```bash
python main_bin.py \
  --batch_size 32 \
  --test_batch_size 32 \
  --dataset vimeo90K_septuplet \
  --loss 1*L1 \
  --max_epoch 200 \
  --lr 0.0002 \
  --data_root input/vimeo_septuplet/ \
  --n_outputs 1 \
  --exp_name binflavr_vimeo2x \
  --checkpoint_dir output \
  --model unet_18 \
  --num_workers 4 # default: 16
```

```bash
python main_bin.py   --batch_size 2   --test_batch_size 2   --dataset vimeo90K_septuplet   --loss 1*L1   --max_epoch 200   --lr 0.0002   --data_root input/vimeo_septuplet/   --n_outputs 1   --exp_name binflavr_vimeo2x_nodata_parallel   --checkpoint_dir output   --model unet_18   --num_workers 2   --cuda True
```

```bash
torchrun --nproc_per_node=4 main_bin_ddp.py   --batch_size 12   --test_batch_size 2   --dataset vimeo90K_septuplet   --loss 1*L1   --max_epoch 200   --lr 0.0002   --data_root input/vimeo_septuplet/   --n_outputs 1   --exp_name binflavr_vimeo2x_batch12   --checkpoint_dir output   --model unet_18   --num_workers 2   --cuda True
```


### GoPro (8x interpolación):

```bash
python main_bin.py \
  --batch_size 16 \
  --dataset gopro \
  --loss 1*L1 \
  --max_epoch 300 \
  --lr 0.0001 \
  --data_root <ruta_a_gopro> \
  --n_outputs 7 \
  --exp_name binflavr_gopro8x
```

---

## 🧪 Evaluación

```bash
python test.py \
  --dataset vimeo90K_septuplet \
  --data_root <ruta_a_datos> \
  --load_from <ruta_a_modelo.pth> \
  --n_outputs 1
```

---

## 🕒 Interpolación de Video Real

```bash
python interpolate.py \
  --input_video <video_entrada.mp4> \
  --factor 8 \
  --load_model <modelo_binarizado.pth>
```

---

## 📊 Evaluación en Middleburry

```bash
python Middleburry_Test.py \
  --data_root <carpeta_middlebury> \
  --load_from <modelo.pth>
```

---

## 🧠 Créditos y Licencia

Este proyecto está basado en el código original de [FLAVR](https://github.com/tarun005/FLAVR), con modificaciones sustanciales para incorporar binarización. Licencia original se mantiene como MIT.

---
