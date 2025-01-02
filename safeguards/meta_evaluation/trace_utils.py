from typing import Any

from weave.trace.refs import ObjectRef, OpRef
from weave.trace.vals import WeaveObject


def serialize_weave_object(obj: WeaveObject):
    serialized_data = obj._val.__dict__
    serialized_data.pop("name", None)
    serialized_data.pop("description", None)
    serialized_data.pop("_class_name", None)
    serialized_data.pop("_bases", None)
    return serialized_data


def serialize_weave_references(data: Any):
    if isinstance(data, ObjectRef):
        return {"type": "ObjectRef", "name": data.name}
    elif isinstance(data, OpRef):
        return {"type": "OpRef", "name": data.name}
    elif isinstance(data, list):
        return [serialize_weave_references(item) for item in data]
    elif isinstance(data, dict):
        return {key: serialize_weave_references(value) for key, value in data.items()}
    else:
        return data


def serialize_input_output_objects(inputs: Any) -> dict[str, Any]:
    inputs = dict(inputs)
    for key, val in inputs.items():
        if isinstance(val, WeaveObject):
            inputs[key] = serialize_weave_object(inputs[key])
        inputs[key] = serialize_weave_references(inputs[key])
    return inputs
