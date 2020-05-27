from openeo_driver.ProcessGraphDeserializer import custom_process
from openeo_driver.save_result import JSONResult


@custom_process
def foobar(args: dict, viewingParameters: dict):
    return JSONResult(data={
        "args": list(args.keys()),
        "msg": "hello world",
    })
