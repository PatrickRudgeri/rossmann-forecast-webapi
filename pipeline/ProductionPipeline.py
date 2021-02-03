import pandas as pd
import numpy as np
import joblib
import math
import inflection 
import datetime as dt

class ProductionPipeline():
	def __init__(self):
		self._load_pkl_objs()

	def _load_pkl_objs(self):
		"""
		Carrega os scalers, encoders e a lista de colunas selecionas na etapa de feature selection. 
		"""
		load = lambda file: joblib.load( "parameter/" + file + ".pkl.bz2")

		attr_names = [
			# scalers
			"competition_distance_scaler",
			"competition_time_month_scaler",
			"promo_time_week_scaler",
			"year_scaler",
			"promo2_since_year_scaler",
			"promo2_since_week_scaler",
			"competition_open_since_year_scaler",

			# encoders
			"state_holiday_encoding",
			"store_type_encoding",
			"assortment_encoding",

			# cols from feature selection
			"cols_selected"
		]

		# Define os atributos da classe com o mesmo nome dos parâmetros salvos 
		for param_name in attr_names:
			setattr(self, param_name, load(f"{param_name}"))


	def data_cleaning(self, df):
		"""
		Renomear colunas, mudança de tipos e tratamento de valores faltantes.
		"""
		df = df.copy()

		## 1.1 - Rename Columns

		cols_old = list(df.columns)

		snakecase = lambda x: inflection.underscore(x)
		df.columns = map(snakecase, cols_old)


		## 1.3 - Change Data Types (pt 1)

		df["date"] = pd.to_datetime(df["date"])


		## 1.5 - Fill out NA

		df['competition_distance'].max()

		input_above_max = lambda x: 200000.0 if math.isnan(x) else x

		df['competition_distance'] = df['competition_distance'].apply(input_above_max)


		input_date_month_if_na = lambda x: x['date'].month if math.isnan(x[
		    'competition_open_since_month']) else x['competition_open_since_month']
		df['competition_open_since_month'] = df.apply(input_date_month_if_na, axis=1)

		input_date_year_if_na = lambda x: x['date'].year if math.isnan(x[
		    'competition_open_since_year']) else x['competition_open_since_year']
		df['competition_open_since_year'] = df.apply(input_date_year_if_na, axis=1)

	
		input_date_week_if_na = lambda x: x['date'].week if math.isnan(x[
		    'promo2_since_week']) else x['promo2_since_week']
		df['promo2_since_week'] = df.apply(input_date_week_if_na, axis=1)

		input_date_year_if_na = lambda x: x['date'].year if math.isnan(x[
		    'promo2_since_year']) else x['promo2_since_year']
		df['promo2_since_year'] = df.apply(input_date_year_if_na, axis=1)

		month_map = {
		    1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
		    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
		}

		df['promo_interval'].fillna(0, inplace=True)

		df['month_map'] = df['date'].dt.month.map(month_map)

		is_promo_func = (lambda x: 0 if x['promo_interval'] == 0 else 1
		                 if x['month_map'] in x['promo_interval'].split(',') else 0)
		df['is_promo'] = df.apply(is_promo_func, axis=1)


		## 1.6 - Change Data types (pt2)

		df['state_holiday'] = df['state_holiday'].astype(str)

		df['competition_open_since_month'] = df[
		    'competition_open_since_month'].astype(int)
		df['competition_open_since_year'] = df['competition_open_since_year'].astype(int)

		df['promo2_since_week'] = df['promo2_since_week'].astype(int)
		df['promo2_since_year'] = df['promo2_since_year'].astype(int)

		return df


	def feature_engineering(self, df):
		""" 
		Cria novas features a partir das features existentes e deleta 
			algumas colunas que não serão úteis. 
		"""

		# year
		df['year'] = df['date'].dt.year

		# month
		df['month'] = df['date'].dt.month

		# day
		df['day'] = df['date'].dt.day

		# week of year
		df['week_of_year'] = df['date'].dt.isocalendar().week

		# year week
		df['year_week'] = df['date'].dt.strftime('%Y-%W')


		# competition since
		combine_year_month = lambda x: (dt.datetime(year=x['competition_open_since_year'],
		                       			month=x['competition_open_since_month'],
		                       			day=1))


		df['competition_since'] = df.apply(combine_year_month, axis=1)
		df['competition_time_month'] = ((df['date'] - df['competition_since']) /
		                                 30).apply(lambda x: x.days).astype(int)

		# promo since
		df['promo_since'] = df['promo2_since_year'].astype(
		    str) + '-' + df['promo2_since_week'].astype(str)
		
		df['promo_since'] = df['promo_since'].apply(lambda x: dt.datetime.strptime(
		    x + '-1', '%Y-%W-%w') - dt.timedelta(days=7))
		
		df['promo_time_week'] = ((df['date'] - df['promo_since']) /
		                          7).apply(lambda x: x.days).astype(int)

		# assortment
		df['assortment'] = df['assortment'].apply(
		    lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')

		# state holiday
		map_state_holiday = (lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' 
			if x == 'b' else 'christmas' if x == 'c' else 'regular_day')
		df['state_holiday'] = df['state_holiday'].apply(map_state_holiday)


		## 3.1 - Filtragem das linhas
		df = df[(df['open'] != 0)]

		# coluna `open` sobrou apenas open==1, logo podemos exclui-la.
		# Dropar colunas que foram usadas para derivar variaveis no processo de feature engineering
		cols_drop = ['id', 'open', 'promo_interval', 'month_map']
		
		return df.drop(cols_drop, axis=1, errors="ignore")


	def data_preparation(self, df):
		""" 
		Reescaling, transformações de natureza, encoding e retorna apenas as 
			colunas selecionadas no processo de feature selection
		"""
		
		def apply_transform(type_, col_name, ohe=False):
			""" Aplicando os scalers e transformações """
			nonlocal df
			parameter = getattr(self, f"{col_name}_{type_}")

			if not ohe:
				df[col_name] = parameter.transform(df[[col_name]].values)
			else:
				tr = parameter.transform(df[[col_name]].values)
				columns = [*map(lambda c: col_name+"_"+c, parameter.categories_[0])]
				df = df.join(pd.DataFrame(tr.toarray(), columns=columns, index=df.index))

		def apply_cyclic_transform(col_name, n):
			"""
			Transforma a coluna `col` do dataframe `df` em uma forma cíclica com seno e cosseno. 
			X * (2 * pi / n) converte os valores para radiano [0, 2*pi]
			`n` é a quantidade de níveis (ex: horas em um dia => n=24)
				[Ref](https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/)
			"""
			nonlocal df
			df[col_name + '_sin'] = np.sin(df[col_name].values * (2 * np.pi / n))
			df[col_name + '_cos'] = np.cos(df[col_name].values * (2 * np.pi / n))				


		features = {
			"scaler": [
				'competition_distance',  
				'competition_time_month',  
				'promo_time_week',  
				'year',  
				'promo2_since_week', 
				'promo2_since_year', 
				'competition_open_since_year'
			],
			"encoding": [
				('state_holiday', True) ,
				('store_type', True),
				'assortment'
			]
		}

		for type_, features_list in features.items():
			for feature in features_list:
				ohe = False
				col_name = feature
				if isinstance(feature, tuple):
					col_name = feature[0]
					ohe = feature[1]

				apply_transform(type_, col_name, ohe)


		### 5.3.3 -Nature Transformation

		# Variáveis que possuem uma natureza ciclica e que devem ser modificadas 
		cyclic_features = [('day_of_week', 7), ('month', 12), 
							('day', 30), ('week_of_year', 52)]
		
		for feature, n in cyclic_features:
			apply_cyclic_transform(feature, n)

		return df[self.cols_selected]


	def get_prediction(self, model, original_data, test_data):
		""" 
		Obtém as predições sobre os dados de test_data e retorna um json 
			com uma nova feature `prediction` nos dados originais.

		"""
		pred = model.predict(test_data.values)

		original_data["prediction"] = np.expm1(pred)
		return original_data.to_json(orient="records", date_format="iso") 


	def start_pipeline(self, raw_df):
		""" 
		Executa o pipeline de pré-processamento e retorna os dados 
			prontos para realizar a modelagem preditiva.
		"""
		prep1 = self.data_cleaning(raw_df)


		prep2 = self.feature_engineering(prep1)


		return self.data_preparation(prep2)