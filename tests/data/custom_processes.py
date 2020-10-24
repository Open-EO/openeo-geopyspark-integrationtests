from openeo_driver.ProcessGraphDeserializer import custom_process
from openeo_driver.save_result import JSONResult
from openeo_driver.utils import EvalEnv


@custom_process
def foobar(args: dict, env: EvalEnv = None):
    return JSONResult(data={
        "args": sorted(args.keys()),
        "msg": "hello world",
    })
