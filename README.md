# Sivil ve Askeri Araç Tespiti

YOLO11 tabanlı bu proje, görüntülerdeki araçları iki sınıfta tespit eder:

- `military_vehicle` — askeri araç
- `civilian_vehicle` — sivil araç

Proje; model eğitimi için betikler ve eğitilmiş ağırlıklarla çalışan Streamlit ve Flask arayüzleri içerir.

## Gereksinimler

- Python 3.10 veya üzeri
- Eğitim için CUDA destekli NVIDIA GPU önerilir; eğitim betiği GPU yoksa CPU kullanır.
- Eğitim verisi ve model ağırlıkları (aşağıdaki dizin yapısına bakın)

## Kurulum

Proje kök dizininde sanal ortam oluşturup etkinleştirin:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Streamlit arayüzü ve eğitim betikleri için:

```powershell
python -m pip install ultralytics streamlit
```

Flask arayüzü için:

```powershell
python -m pip install -r webapp\requirements.txt
```

PyTorch/CUDA kurulumu donanım ve CUDA sürümüne göre değişebilir. GPU ile eğitim yapacaksanız PyTorch'un resmi kurulum sayfasından sisteminize uygun komutu seçin. CPU ile çalıştırmak için de PyTorch kurulu olmalıdır.

## Veri seti

`dataset.yaml`, YOLO veri setinin `datasets/merged` altında şu yapıda olduğunu varsayar:

```text
datasets/merged/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Etiket dosyaları YOLO biçiminde olmalı ve sınıf kimlikleri `0` (askeri araç) ile `1` (sivil araç) olmalıdır. Bu depoda ham ve işlenmiş veri setlerinin bulunması garanti edilmez. Veri setlerini edinirken lisans ve yeniden dağıtım koşullarını ayrıca kontrol edin.

Depodaki veri hazırlama betikleri farklı yerel veri kaynaklarını bekler. Örneğin `convert_nwpu.py`, `nwpu vhr10 dataset/` dizinini; `merge_dataset.py` ise `military-vehicle.v6i.yolov11/` ve dönüştürülmüş NWPU verisini kullanır. Betikleri çalıştırmadan önce kaynak dizinlerini ve hedef yolları kendi verinize göre düzenleyin.

## Model eğitimi

Veri seti `dataset.yaml` içindeki yollarla hazırlandıktan sonra proje kökünde:

```powershell
python train.py
```

Eğitim, Ultralytics YOLO11l başlangıç modelini kullanır ve sonuçları `runs/train/military_or_not/` altında oluşturur. En iyi ağırlık dosyası varsayılan olarak `runs/train/military_or_not/weights/best.pt` konumuna kaydedilir. Eğitim ayarları `train.py` içindedir.

## Görüntü üzerinde tahmin

### Streamlit

```powershell
streamlit run app.py
```

Uygulamayı tarayıcıda açın, JPG/JPEG/PNG görseli yükleyin ve tespitleri görüntüleyin.

### Flask

```powershell
python webapp\app.py
```

Ardından `http://127.0.0.1:5000` adresini açın. Görsel yükleme sınırı 15 MB'tır; sonuç görseli ve tespit bilgileri arayüzde gösterilir.

Her iki arayüz de varsayılan olarak `runs/train/military_or_not/weights/best.pt` dosyasını kullanır. Farklı bir model dosyası belirtmek için `MODEL_PATH` ortam değişkenini ayarlayın:

```powershell
$env:MODEL_PATH = "C:\models\best.pt"
streamlit run app.py
```

Flask uygulamasını başlatırken de aynı ortam değişkeni kullanılabilir.

## Yardımcı betikler

- `map_metrics.py`: Eğitilmiş model için Ultralytics doğrulamasını çalıştırır ve mAP metriklerini yazdırır.
- `check_labels.py`: Belirli bir veri dizinindeki etiket sınıflarını sayar.
- `convert_nwpu.py`, `merge_dataset.py`, `split_balance_val.py`, `update_mvrsd_labels.py`: Veri dönüştürme, birleştirme ve bölme işlemleri için yardımcı betiklerdir; kaynak yolları betiklerin içinde tanımlanmıştır.
- `fp_fn.py`: Tahmin örnekleri oluşturmak için kullanılır; veri ve model yollarını çalıştırmadan önce kontrol edin.

## GitHub'a yükleme notu

Veri setleri, model ağırlıkları, eğitim çıktıları ve yüklenen görseller büyük veya yeniden dağıtımı kısıtlı olabilir. Bunlar `.gitignore` ile yeni eklemelerden hariç tutulur. Ancak daha önce Git geçmişine eklenmiş dosyaları `.gitignore` kaldırmaz. GitHub'a göndermeden önce dosyaların lisansını ve Git geçmişinin boyutunu kontrol edin.
