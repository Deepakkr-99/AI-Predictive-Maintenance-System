def failure_reason(tool_wear, torque, speed):

    reason = []

    if tool_wear > 200:
        reason.append("High Tool Wear")

    if torque > 60:
        reason.append("High Torque")

    # speed 0 hone par low speed reason mat dikhao
    if speed > 0 and speed < 1300:
        reason.append("Low Rotational Speed")

    if len(reason) == 0:
        reason.append("Machine Parameters are Normal")

    return reason
