# 🧠 Beyin İnme Tespit ve Segmentasyon Sistemi

Bu proje, tıbbi görüntüleme verilerinden beyin inme bölgelerini tespit etmek ve segment etmek için geliştirilmiş kapsamlı bir deep learning sistemidir. MONAI framework'ü ve 3D Slicer entegrasyonu ile profesyonel tıbbi görüntü analizi sunar.

## ✨ Ana Özellikler

### 🔬 3D Slicer Extension
- **Tam entegre extension**: 3D Slicer içinde çalışan profesyonel modül
- **Gerçek zamanlı analiz**: Anında sınıflandırma ve segmentasyon
- **3D görselleştirme**: İnme bölgelerinin 3 boyutlu görüntülenmesi
- **Batch işleme**: Tüm dataset üzerinde otomatik analiz
- **Kullanıcı dostu arayüz**: Qt tabanlı GUI

### 🧠 Beyin Çıkarma (Brain Extraction)
- **Otomatik kafatası segmentasyonu**: Akıllı algoritma ile beyin dokusunun çıkarılması
- **Morfolojik işlemler**: Gelişmiş görüntü işleme teknikleri
- **Boşluk doldurma**: Segmentasyon maskelerindeki boşlukların otomatik doldurulması

### 🤖 Deep Learning Modelleri
- **Sınıflandırma**: DenseNet121 ile binary inme tespiti
- **Segmentasyon**: AttentionUnet ile inme bölgesi lokalizasyonu
- **Pre-trained modeller**: Hazır kullanıma hazır eğitilmiş ağırlıklar

### 📊 Analiz ve Değerlendirme
- **Detaylı metrikler**: Dice skoru, confusion matrix, accuracy
- **Slice-by-slice analiz**: Her kesit için ayrı değerlendirme

## 🛠️ Kullanılan Teknolojiler

- **[MONAI](https://monai.io/)**: Tıbbi görüntüleme için deep learning framework
- **PyTorch**: Neural network backend
- **3D Slicer**: Tıbbi görüntüleme yazılımı entegrasyonu
- **OpenCV**: Görüntü işleme
- **NumPy**: Sayısal hesaplamalar

## 📁 Proje Yapısı

```
stroke-detection-models/
├── 3d_slicer_extension/          # 3D Slicer Extension
│   └── StrokeDetection/
│       ├── StrokeDetectionModule/
│       │   ├── Resources/
│       │   │   ├── Icons/        # UI ikonları
│       │   │   ├── Models/       # Pre-trained modeller
│       │   │   │   ├── classification_model.pth
│       │   │   │   └── segmentation_model.pth
│       │   │   └── UI/           # Qt UI dosyaları
│       │   ├── StrokeDetectionModule.py  # Ana modül
│       │   └── StrokeModels/
│       │       ├── classification.py    # Sınıflandırma modeli
│       │       └── segmentation.py      # Segmentasyon modeli
│       └── CMakeLists.txt        # Build yapılandırması
├── notebooks/                    # Jupyter Notebooks
│   ├── brain_extraction/         # Beyin çıkarma algoritmaları
│   ├── classification/           # Sınıflandırma deneyleri
│   │   ├── 01_raw_patchless.ipynb
│   │   ├── 02_raw_patch.ipynb
│   │   └── 03_brain_extraction_patchless.ipynb
│   └── segmentation/             # Segmentasyon deneyleri
│       ├── 01_synthetic_image_segmentation.ipynb
│       ├── 02_raw_patchless.ipynb
│       └── 03_brain_extraction_patchless.ipynb
├── data/                         # Veri dosyaları
└── LICENSE                       # MIT Lisansı
```

## 🚀 Kurulum

### 3D Slicer Extension Kurulumu

1. **3D Slicer'ı indirin**: [3D Slicer Official Website](https://www.slicer.org/)
2. **Extension'ı yükleyin**:
   ```bash
   git clone https://github.com/your-repo/stroke-detection-models.git
   cd stroke-detection-models/3d_slicer_extension
   ```
3. **3D Slicer'da extension'ı aktifleştirin**

### 3D Slicer Python Environment Kurulumu

3D Slicer kendi Python ortamını kullandığı için, gerekli kütüphanelerin Slicer'ın Python konsolu üzerinden yüklenmesi gerekir:

#### 3D Slicer İçinde Kurulum
1. **3D Slicer'ı açın**
2. **Python Console'u açın** (View → Python Interactor)
3. **Gerekli paketleri yükleyin**:

```python
# 3D Slicer Python Console'da çalıştırın
import subprocess
import sys

# MONAI ve bağımlılıklarını yükle
subprocess.check_call([sys.executable, "-m", "pip", "install", "monai"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

print("Kütüphaneler başarıyla yüklendi!")
```

#### Notebook Geliştirme için Ayrı Environment (Opsiyonel)

Notebook'ları geliştirmek için ayrı bir environment de kurabilirsiniz:

```bash
# Conda environment oluşturun
conda create -n stroke-notebooks python=3.8
conda activate stroke-notebooks

# Notebook geliştirme için paketler
pip install torch torchvision
pip install monai
pip install opencv-python
pip install jupyter
pip install matplotlib
pip install nibabel
```

#### Kurulum Doğrulama

3D Slicer Python Console'da test edin:
```python
# Kütüphanelerin yüklendiğini doğrula
import torch
import monai
import cv2
print("Tüm kütüphaneler başarıyla yüklendi!")
```

## 📦 Örnek Test Verileri

Modelleri test etmek için [AISD (Acute Ischemic Stroke Dataset)](https://github.com/GriffinLiang/AISD) kullanabilirsiniz:

### Veri İndirme
1. **Image verilerini indirin**: [image.zip] - Baidu YUN (şifre: "aisd") veya Google Drive
2. **Mask verilerini indirin**: [mask.zip] - Baidu YUN (şifre: "aisd") veya Google Drive

### Veri Formatı Dönüştürme
- **PNG'den NII.GZ'ye**: `Png2NiiGz.ipynb` notebook dosyasındaki fonksiyonları kullanın
- **3D Slicer uyumluluğu**: .nii.gz formatına çevrilmiş veriler doğrudan yüklenebilir


## 🎯 Kullanım

### 3D Slicer Extension Kullanımı

#### Tekil Analiz
1. **3D Slicer'ı açın**
2. **StrokeDetection modülünü seçin**
3. **CT/MRI görüntüsünü yükleyin** (AISD'den .nii.gz formatında)
4. **Ground truth mask'ını yükleyin** (opsiyonel)
5. **Classification** butonuna tıklayın
6. **Segmentation** butonuna tıklayın (classification sonrası)
7. **3D visualization** aktifleştirin

#### Batch İşleme (Dataset Analizi)
**Klasör Yapısı Gereksinimleri:**
```
dataset_folder/
├── image/
│   ├── patient001.nii.gz
│   ├── patient002.nii.gz
│   └── ...
└── mask/
    ├── patient001.nii.gz
    ├── patient002.nii.gz
    └── ...
```

**Kullanım Adımları:**
1. **Dataset klasörünü seçin** (yukarıdaki yapıya uygun)
2. **"Classify Dataset"** veya **"Segment Dataset"** butonuna tıklayın
3. **Otomatik işleme** başlar ve sonuçlar HTML tablosunda görüntülenir
4. **İlerleme durumu** real-time olarak takip edilir

**Önemli Notlar:**
- `image/` ve `mask/` klasörlerinde **aynı isimde** .nii.gz dosyaları bulunmalı
- Dosya isimleri **tam olarak eşleşmelidir** (patient001.nii.gz ↔ patient001.nii.gz)
- Batch işleme sırasında **tüm dataset** otomatik olarak analiz edilir

### Notebook Kullanımı

```bash
jupyter notebook
# notebooks/ klasöründeki ilgili notebook'u açın
# AISD verilerini test etmek için Png2NiiGz.ipynb'yi kullanın
```

## 📈 Model Performansı

- **Classification Model**: DenseNet121 - Binary inme tespiti
- **Segmentation Model**: AttentionUnet - İnme bölgesi segmentasyonu
- **Brain Extraction**: Morfolojik işlemler ile %95+ doğruluk

## 🔬 Araştırma Yaklaşımları

### Sınıflandırma Stratejileri
- **Patch-based training**: Görüntü parçaları ile eğitim
- **Full image training**: Tam görüntü ile eğitim
- **Brain extraction**: Ön işleme ile performans artırımı

### Segmentasyon Teknikleri
- **Attention mechanism**: Önemli bölgelere odaklanma
- **Multi-scale features**: Farklı ölçeklerde özellik çıkarma
- **Post-processing**: Morfolojik iyileştirmeler

## 📝 Değerlendirme Metrikleri

- **Dice Score**: Segmentasyon overlap'i
- **Accuracy**: Sınıflandırma doğruluğu
- **Sensitivity/Specificity**: Detaylı performans analizi
- **Confusion Matrix**: Hata analizi

## 🤝 Katkıda Bulunma

1. Bu repository'yi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

**Metin Ilgar Mutlu**

---

> **Bu proje, biyomedikal mühendisliği dersi kapsamında geliştirilmiştir ve gerçek tıbbi uygulamalarda kullanım için ek validasyon gerektirebilir.**
