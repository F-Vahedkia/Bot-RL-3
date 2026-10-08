
# مراحل اضافه کردن هر لایه جدید (مثلاً f04_features):


# ---1 --- در _init_components:
from f04_features.features_engine import FeaturesEngine

self.features_engine = FeaturesEngine(self.cfg)
self._components["features"] = self.features_engine



# --- 2 --- در حلقه اصلی (run_live_loop):
features = self.features_engine.calculate(self.data_handler._cached_df)