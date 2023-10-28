import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import List, Tuple
from typing import Union

class DelayModel:
    def __init__(self):
        self._model = None

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        if target_column is not None:
            # Separar las características y el objetivo
            features = data.drop(columns=[target_column])
            target = data[target_column]
            return features, target
        else:
            # Si no se proporciona una columna de destino, devolver solo características
            return data

    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        self._model = XGBClassifier(random_state=1, learning_rate=0.01)
        self._model.fit(features, target)

    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        return predictions

def main():
    # Cargar datos desde 'tu_archivo.csv'
    data = pd.read_csv('../data/data.csv')

    # Eliminar columnas que no se van a utilizar
    columns_to_drop = ['Fecha-I', 'Vlo-I', 'Fecha-O', 'Vlo-O', 'DIANOM']
    data = data.drop(columns=columns_to_drop)
        
    print(data.columns)

    model = DelayModel()

    # Define el nombre de la columna objetivo
    target_column = 'delay'  # Reemplaza 'TU_COLUMNA_OBJETIVO' con el nombre de tu columna objetivo

    # Codificar columnas categóricas utilizando one-hot encoding
    categorical_columns = ['Ori-I', 'Des-I', 'Emp-I', 'Ori-O', 'Des-O', 'Emp-O', 'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES', 'period_day', 'high_season']
    data = pd.get_dummies(data, columns=categorical_columns)

    # Asegurarse de que la columna 'DIANOM' existe en el DataFrame actual
    if 'DIANOM' in data:
        # Convertir los valores de días de la semana en números
        data['DIANOM'] = data['DIANOM'].map({'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'Jueves': 3, 'Viernes': 4, 'Sábado': 5, 'Domingo': 6})

    # Preprocesar los datos y obtener las características y el objetivo
    features, target = model.preprocess(data, target_column)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

    # Entrenar el modelo
    model.fit(x_train, y_train)

    # Hacer predicciones en el conjunto de prueba
    predictions = model.predict(x_test)

    # Imprimir las predicciones
    print(predictions)

if __name__ == "__main__":
    main()

