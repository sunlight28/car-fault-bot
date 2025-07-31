import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 50


FAULT_DESCRIPTIONS = {
    'battery': {
        'description': 'Проблемы с аккумуляторной батареей',
        'recommendations': [
            '1. Диагностика:',
            '   - Измерьте напряжение на клеммах (норма: 12.6В при выключенном двигателе)',
            '   - Проверьте плотность электролита (для обслуживаемых АКБ)',
            '   - Осмотрите клеммы на окисление',
            '2. Возможные решения:',
            '   - Очистите клеммы от окислов (используйте раствор соды)',
            '   - Плотно затяните клеммы',
            '   - Если напряжение ниже 11.8В - требуется зарядка',
            '3. При частых разрядках проверьте:',
            '   - Работу генератора',
            '   - Утечки тока в бортовой сети'
        ],
        'warning': '⚠️ Не замыкайте клеммы инструментами! Это может вызвать взрыв АКБ.'
    },
    'tire': {
        'description': 'Проблемы с шинами',
        'recommendations': [
            '1. Проверьте:',
            '   - Давление в шинах (сравните с рекомендованным для вашей модели)',
            '   - Глубину протектора (минимум 1.6 мм для легковых авто)',
            '   - Боковые порезы и грыжи',
            '2. Для разных случаев:',
            '   - Медленный спуск: ищите гвоздь/саморез в протекторе',
            '   - Боковой разрыв: замените шину (ремонт небезопасен)',
            '   - Износ с одной стороны: проверьте развал-схождение',
            '3. Экстренные меры:',
            '   - Установите запаску или используйте ремкомплект',
            '   - Не превышайте скорость 80 км/ч на докатке'
        ],
        'warning': '❗ Езда на спущенной шине повреждает боковину! Остановитесь сразу при обнаружении.'
    },
    'engine_oil_leak': {
        'description': 'Утечка моторного масла',
        'recommendations': [
            '1. Локализация проблемы:',
            '   - Прокладка клапанной крышки (масло на верхней части двигателя)',
            '   - Сальники коленвала (масло снизу возле шкивов)',
            '   - Масляный фильтр или поддон (лужи под машиной после стоянки)',
            '2. Временные меры:',
            '   - Долейте масло до уровня между метками на щупе',
            '   - Используйте герметик для прокладок (как временное решение)',
            '3. Серьёзные случаи:',
            '   - При сильной течи заглушите двигатель',
            '   - Вызовите эвакуатор при уровне масла ниже минимума'
        ],
        'warning': '⚠️ Низкий уровень масла может привести к заклиниванию двигателя!'
    },
    'engine_black_smoke': {
        'description': 'Чёрный дым из выхлопной трубы',
        'recommendations': [
            '1. Основные причины:',
            '   - Переобогащение топливной смеси',
            '   - Неисправность топливных форсунок или ТНВД (для дизелей)',
            '   - Забит воздушный фильтр',
            '   - Проблемы с датчиком кислорода (лямбда-зондом)',
            '2. Проверьте:',
            '   - Состояние воздушного фильтра',
            '   - Давление топлива',
            '   - Показания датчиков OBD-2 (если есть сканер)',
            '3. Для дизельных двигателей:',
            '   - Возможен износ турбокомпрессора',
            '   - Проверьте систему EGR'
        ],
        'warning': '⚠️ Чёрный дым часто сопровождается повышенным расходом топлива!'
    },
    'normal': {
        'description': 'Очевидных неисправностей не обнаружено',
        'recommendations': [
            '1. Рекомендуемые действия:',
            '   - Продолжайте регулярное ТО согласно регламенту',
            '   - Следите за уровнем технических жидкостей',
            '   - Обращайте внимание на новые необычные звуки/вибрации',
            '2. Для профилактики:',
            '   - Проверяйте давление в шинах раз в 2 недели',
            '   - Осматривайте подкапотное пространство на наличие подтёков',
            '   - Следите за показаниями приборной панели'
        ],
        'warning': 'ℹ️ При появлении симптомов сделайте повторную диагностику.'
    }
}


class CarFaultDetector:
    def __init__(self):
        self.embedding_model = self.build_embedding_model()
        self.classifier = None
        self.class_names = list(FAULT_DESCRIPTIONS.keys())

    def build_embedding_model(self):
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3),
            pooling='avg'
        )
        return Model(inputs=base_model.input, outputs=base_model.output)

    def extract_embeddings(self, data_dir):
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        generator = datagen.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False
        )

        embeddings = []
        labels = []
        for _ in range(len(generator)):
            x, y = next(generator)
            emb = self.embedding_model.predict(x, verbose=0)
            embeddings.extend(emb)
            labels.extend(y)

        return np.array(embeddings), np.array(labels)

    def train_classifier(self, data_dir):
        X, y = self.extract_embeddings(data_dir)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

        self.classifier = SVC(kernel='rbf', C=10, probability=True)

        self.classifier.fit(X_train, y_train)

        self.embedding_model.save('embedding_model.keras')
        joblib.dump(self.classifier, 'svm_classifier.joblib')

        val_acc = self.classifier.score(X_val, y_val)
        print(f"Validation accuracy: {val_acc:.1%}")
        return val_acc


    def load_models(self):
        if os.path.exists('embedding_model.keras') and os.path.exists('svm_classifier.joblib'):
            self.embedding_model = tf.keras.models.load_model('embedding_model.keras')
            self.classifier = joblib.load('svm_classifier.joblib')
            return True
        return False


    def predict_fault(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB').resize(IMAGE_SIZE)
            img_array = preprocess_input(np.array(img))
            img_array = np.expand_dims(img_array, axis=0)

            embedding = self.embedding_model.predict(img_array, verbose=0)
            proba = self.classifier.predict_proba(embedding)[0]

            pred_class_idx = np.argmax(proba)
            predicted_class = self.class_names[pred_class_idx]
            confidence = proba[pred_class_idx]

            return predicted_class, confidence
        except Exception as e:
            print(f"Ошибка при обработке изображения: {e}")
            return None, 0.0



def main():
    print("=" * 50)
    print("Чат-бот для диагностики автомобильных неисправностей")
    print("=" * 50)

    detector = CarFaultDetector()

    if detector.load_models():
        print("Модели загружены успешно!")
    else:
        print("Обученные модели не найдены. Необходимо обучить модель.")
        dataset_path = input("Введите путь к папке с данными для обучения (или нажмите Enter для пропуска): ")
        if dataset_path and os.path.exists(dataset_path):
            print("Извлекаем эмбеддинги и обучаем классификатор...")
            detector.train_classifier(dataset_path)
            print("Обучение завершено!")
        else:
            print("Продолжаем без обучения (точность будет низкой)")

    while True:
        print("\n1. Диагностика по фото")
        print("2. Выход")
        choice = input("Выберите действие: ")

        if choice == '1':
            image_path = input("Введите путь к изображению: ").strip()
            if not os.path.exists(image_path):
                print("Файл не найден!")
                continue

            fault, confidence = detector.predict_fault(image_path)
            if fault:
                info = FAULT_DESCRIPTIONS.get(fault, FAULT_DESCRIPTIONS['normal'])
                print(f"\nРезультат: {info['description']} (точность: {confidence:.1%})")
                print("Рекомендации:")
                for i, rec in enumerate(info['recommendations'], 1):
                    print(f"{rec}")
                if 'warning' in info:
                    print(f"\n{info['warning']}")
        elif choice == '2':
            print("До свидания!")
            break
        else:
            print("Некорректный ввод")


if __name__ == "__main__":
    main()