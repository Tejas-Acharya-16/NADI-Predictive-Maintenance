# FD001 data audit

FD001 is simulated turbofan telemetry, not a Siemens pump, truck, compressor, or motor-vibration dataset. The training file contains 20,631 rows, 100 engines, and 26 raw columns. Checks found zero missing/non-finite cells, duplicate unit-cycle records, sorted-cycle faults, cycle gaps, or invalid IDs.

## Feature audit

The existing causal builder intentionally includes the current cycle and `cycles_seen`; it excludes engine ID. The grouped study adds a cycle-free ablation so this available-at-inference signal is measured rather than silently assumed. Near-constant training sensors at standard deviation below 0.01: sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19. Sensor removal is fit within each development fold only.

## Sensor ranges

```text
                 min        max        std
sensor_1    518.6700   518.6700   0.000000
sensor_2    641.2100   644.5300   0.500053
sensor_3   1571.0400  1616.9100   6.131150
sensor_4   1382.2500  1441.4900   9.000605
sensor_5     14.6200    14.6200   0.000000
sensor_6     21.6000    21.6100   0.001389
sensor_7    549.8500   556.0600   0.885092
sensor_8   2387.9000  2388.5600   0.070985
sensor_9   9021.7300  9244.5900  22.082880
sensor_10     1.3000     1.3000   0.000000
sensor_11    46.8500    48.5300   0.267087
sensor_12   518.6900   523.3800   0.737553
sensor_13  2387.8800  2388.5600   0.071919
sensor_14  8099.9400  8293.7200  19.076176
sensor_15     8.3249     8.5848   0.037505
sensor_16     0.0300     0.0300   0.000000
sensor_17   388.0000   400.0000   1.548763
sensor_18  2388.0000  2388.0000   0.000000
sensor_19   100.0000   100.0000   0.000000
sensor_20    38.1400    39.4300   0.180746
sensor_21    22.8942    23.6184   0.108251
```
