# mox_drift_sim/data_schema.py
"""
定义合成 MOx 漂移数据的 schema。

很多生成器代码只需要一个名字为 SIMULATION_SCHEMA 的对象，
用来统一列名和数据类型。这里用一个简单的 dict 来表达。
"""

# 一个最小可用的 schema，后续如果 Data-Generator.md 里有更详细版本，
# 可以再替换成更复杂的结构（例如 pydantic / dataclass）。
SIMULATION_SCHEMA = {
    "time": "datetime64[ns]",   # 时间戳
    "phase": "category",        # WARMUP / FIELD 等
    "mode": "category",         # operating mode / scenario
    "sensor_value": "float32",  # 传感器输出
    "temperature": "float32",   # 环境温度
    "humidity": "float32",      # 环境湿度
    "gas_1_ppm": "float32",     # 有效气体浓度
    "drift_flag": "int8",       # 漂移 / 异常标记（0/1）
}
