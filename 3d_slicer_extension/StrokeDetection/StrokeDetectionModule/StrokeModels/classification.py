"""
Classification model for stroke detection
"""

import os
import numpy as np
import torch
import cv2
from monai.transforms import Compose, Resize, ScaleIntensity, ToTensor
from monai.networks.nets import DenseNet121


def segment_brain_inside_skull(image_data, skull_threshold=254, brain_threshold=10, max_regions=3, min_region_size=1800):
    """
    Segment brain inside skull and fill gaps in segmentation regions
    
    Parameters:
    -----------
    image_data : numpy.ndarray
        Input image data
    skull_threshold : int, optional
        Threshold for skull detection, defaults to 254
    brain_threshold : int, optional
        Threshold for brain detection, defaults to 10
    max_regions : int, optional
        Maximum number of regions to consider, defaults to 3
    min_region_size : int, optional
        Minimum region size to consider, defaults to 1800
        
    Returns:
    --------
    tuple
        (original_image, skull_cavity_mask, final_brain_mask, brain_only, colored_result, selected_regions)
    """
    # Ensure the image is in grayscale
    if len(image_data.shape) > 2:
        image = np.mean(image_data, axis=2).astype(np.uint8) if image_data.shape[2] > 1 else image_data[:, :, 0].astype(np.uint8)
    else:
        image = image_data.astype(np.uint8)

    # 1. Kafatası konturu belirleme
    # Önce görüntüyü yumuşatma
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Kafatasının yüksek yoğunluklu bölgelerini tespit et
    _, skull_thresh = cv2.threshold(blurred, skull_threshold, 255, cv2.THRESH_BINARY)

    # Morfolojik işlemlerle kafatasını kalınlaştır
    kernel = np.ones((5, 5), np.uint8)
    skull_dilated = cv2.dilate(skull_thresh, kernel, iterations=2)

    # 2. Kafatası konturunun içini doldur (kafatası boşluğunu elde et)
    # Önce büyük bir kontur elde etmek için morfolojik işlemlerle kapatma
    skull_closed = cv2.morphologyEx(skull_dilated, cv2.MORPH_CLOSE, kernel, iterations=10)

    # Kafatası boşluğunu içerecek şekilde tüm konturları bul
    contours, _ = cv2.findContours(skull_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Kafatası konturu için boş bir maske oluştur
    skull_cavity_mask = np.zeros_like(image)

    # En büyük konturu kafatası olarak varsay ve içini doldur
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        # Convex Hull ile konturu tamamla
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(skull_cavity_mask, [hull], 0, 255, -1)

    # 3. Beyin bölgesini tespit et
    # Kafatası içindeki bölgelere odaklan (kafatası olmayan bölgeyi al)
    non_skull = cv2.bitwise_not(skull_dilated)

    # Kafatası boşluğu VE kafatası olmayan bölge = sadece kafatası içindeki yumuşak doku
    inner_region = cv2.bitwise_and(skull_cavity_mask, non_skull)

    # Beyin dokusunu diğer yumuşak dokulardan ayırmak için eşikleme
    _, brain_thresh = cv2.threshold(cv2.bitwise_and(blurred, blurred, mask=inner_region),
                                  brain_threshold, 255, cv2.THRESH_BINARY)

    # Morfolojik işlemlerle beyin bölgesini temizle
    brain_mask = cv2.morphologyEx(brain_thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # 4. Büyük bağlantılı beyin bölgelerini seç
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(brain_mask, connectivity=8)

    # Sadece yeterince büyük (min_region_size'dan büyük) bölgeleri filtrele
    filtered_regions = []
    for i in range(1, num_labels):  # 0 indeksi arka plan olduğu için atla
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_region_size:
            filtered_regions.append((i, area))

    # Alan bazında büyükten küçüğe sırala
    filtered_regions.sort(key=lambda x: x[1], reverse=True)

    # En fazla max_regions sayıda bölgeyi seç
    selected_regions = filtered_regions[:max_regions]

    # Seçilen bölgelerin maskelerini hazırla
    final_brain_mask = np.zeros_like(brain_mask)

    # Her bir seçilen bölgeyi ayrı ayrı işle
    for label, _ in selected_regions:
        # Bölgeyi maskele
        current_mask = (labels == label).astype(np.uint8) * 255

        # Bölgedeki boşlukları doldur
        # Konturları bul (dış kontur ve iç boşluklar)
        contours, hierarchy = cv2.findContours(current_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Tüm konturları (dış kontur ve tüm iç boşluklar) doldur
        filled_mask = np.zeros_like(current_mask)
        for i, contour in enumerate(contours):
            # hierarchy[0, i, 3] == -1 ise bu dış kontur demektir
            # Tüm konturları doldurmak istediğimiz için bu kontrolü kullanmıyoruz
            cv2.drawContours(filled_mask, [contour], 0, 255, -1)

        # Doldurulmuş maske ile final maskeyi güncelle
        final_brain_mask = cv2.bitwise_or(final_brain_mask, filled_mask)

    # Boşlukları doldurulmuş maskeyi bir daha işleyelim
    # Alternatif bir yaklaşımla, morfolojik kapatma (closing) işlemi de uygulayalım
    final_brain_mask = cv2.morphologyEx(final_brain_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # Beyin bölgesini çıkar
    brain_only = cv2.bitwise_and(image, image, mask=final_brain_mask)

    # Görselleştirme için renkli sonuç hazırla
    colored_result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Renkli görselleştirmede her bölgeyi farklı renkle işaretle
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255)]  # Yeşil, Sarı, Mor

    # Farklı renklerle segmentasyon görselleştirmesi için tekrar bağlantılı bileşen analizi yap
    # (boşluklar doldurulduktan sonra)
    num_labels_filled, labels_filled = cv2.connectedComponents(final_brain_mask)

    # Her bir seçilen bölge için farklı renk kullan
    for i, (original_label, _) in enumerate(selected_regions):
        if i < len(colors):
            # Doldurulmuş bölgelerdeki yeni etiketi bul
            # En çok örtüşen bölgeyi bul
            max_overlap = 0
            max_overlap_label = 0

            for j in range(1, num_labels_filled):
                # Orijinal maske
                original_mask = (labels == original_label)
                # Yeni maske
                new_mask = (labels_filled == j)
                # Örtüşen pikseller
                overlap = np.sum(np.logical_and(original_mask, new_mask))

                if overlap > max_overlap:
                    max_overlap = overlap
                    max_overlap_label = j

            # En çok örtüşen bölgeyi renklendir
            if max_overlap_label > 0:
                region_mask = (labels_filled == max_overlap_label)
                colored_result[region_mask] = colors[i]

    # Kafatasını kırmızı olarak işaretle
    colored_result[skull_dilated > 0] = [0, 0, 255]

    # Hem seçilen bölgeleri hem de beyin bölgelerinin boyut sıralamasını döndür
    return image, skull_cavity_mask, final_brain_mask, brain_only, colored_result, selected_regions


class StrokeClassificationModel:
    """
    Class for handling stroke classification using DenseNet121
    """
    def __init__(self, model_path):
        """
        Initialize the stroke classification model
        
        Parameters:
        -----------
        model_path : str
            Path to the model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=2).to(self.device)
        
        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Class labels
        self.class_labels = ["Sağlıklı", "Sağlıksız"]
        
        # Transform for processing images including brain extraction and patch extraction
        self.transform = Compose([
            Resize(spatial_size=(512, 512)),
            ScaleIntensity(),
            ToTensor(),
        ])
    
    def preprocess_image(self, image_data):
        """
        Preprocess the input image data with brain extraction
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Raw image data
            
        Returns:
        --------
        torch.Tensor
            Processed image tensor
        """
        # Extract brain from the image
        _, _, _, brain_only, _, _ = segment_brain_inside_skull(image_data)
        
        # Add channel dimension if needed
        if len(brain_only.shape) == 2:
            brain_only = np.expand_dims(brain_only, axis=0)
        else:
            # If it has multiple channels, convert to single channel
            brain_only = np.expand_dims(np.mean(brain_only, axis=2) if brain_only.shape[2] > 1 else brain_only[:, :, 0], axis=0)
        
        # Apply transforms (resize, CLAHE, scale intensity)
        image_tensor = self.transform(brain_only.astype(np.float32))
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image_data):
        """
        Classify the input image
        
        Parameters:
        -----------
        image_data : numpy.ndarray
            Input image data
            
        Returns:
        --------
        dict
            Dictionary containing prediction results
        """
        image_tensor = self.preprocess_image(image_data)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            stroke_probability = probabilities[0][1].item()  # Probability of stroke class
            
        return {
            "predicted_class": predicted_class,
            "class_name": self.class_labels[predicted_class],
            "probabilities": probabilities[0].cpu().numpy(),
            "stroke_probability": stroke_probability
        }