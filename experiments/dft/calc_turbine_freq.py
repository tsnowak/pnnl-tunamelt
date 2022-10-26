def rpm_to_blade_rotation_per_second(rpm, n_blades=3):
    return rpm * n_blades / 60.0


# reference - https://link.springer.com/content/pdf/10.1007/978-3-540-77932-2.pdf
# pg. 126 - 100 rpm
# pg. 211 - 17 rpm
# pg. 215 - 50 rpm
# pg. 218 - 32 rpm

# Rpm
rpm_range = [17.0, 100.0]
n_blades = 3
bps_range = [
    rpm_to_blade_rotation_per_second(rpm_range[0], n_blades=n_blades),
    rpm_to_blade_rotation_per_second(rpm_range[1], n_blades=n_blades),
]
sample_rate = 10
print(f"RPM:{rpm_range}\nBlade per second:{bps_range}")

is_valid = []
for bps in bps_range:
    is_valid.append((bps <= (sample_rate / 2.0)))
if is_valid.count(False) == 0:
    print(
        f"BPS is always less than Nyquist Frequency of camera {10/2} (assuming # blade > {n_blades})."
    )
else:
    print(
        f"BPS may be greater than Nyquist Frequency of camera {10/2} (assuming # blade > {n_blades})."
    )
