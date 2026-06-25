%% Odometry Static Analysis
clc; clear all; close all;

% ---- Load data ----
load('Data/ego.mat')
est = log.estimation;

t        = est.stamp__tot;
t0       = t - t(1);
vx       = est.vx;
vy       = est.vy;
ax       = est.ax;
ay       = est.ay;
heading  = est.ekf_heading;
yaw_rate = est.yaw_rate;
px       = est.gps_pos_report__x(:,1);
py       = est.gps_pos_report__y(:,1);

% ---- Compute Magnitudes & Conversions ----
spd  = sqrt(vx.^2 + vy.^2);
acc  = sqrt(ax.^2 + ay.^2);
hdg  = mod(rad2deg(heading),360);
yr   = rad2deg(yaw_rate);

%% ---- Figure 1: Kinematics & Trajectory ----
figure('Name', 'Ego Odometry Analysis', 'Color', 'w');

% 1. Velocity Plot
subplot(2,3,1);
plot(t0, vx, 'LineWidth', 1.2); hold on;
plot(t0, vy, 'LineWidth', 1.2);
plot(t0, spd, 'k', 'LineWidth', 1.5);
title('Velocity Profile'); xlabel('Time [s]'); ylabel('[m/s]');
legend('Vx', 'Vy', '|V|', 'Location', 'best');
grid on;

% 2. Acceleration Plot
subplot(2,3,2);
plot(t0, ax, 'LineWidth', 1.2); hold on;
plot(t0, ay, 'LineWidth', 1.2);
plot(t0, acc, 'r', 'LineWidth', 1.5);
title('Acceleration Profile'); xlabel('Time [s]'); ylabel('[m/s²]');
legend('Ax', 'Ay', '|A|', 'Location', 'best');
grid on;

% 3. Heading Plot
subplot(2,3,3);
plot(t0, hdg, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]);
title('Heading Angle'); xlabel('Time [s]'); ylabel('[deg]');
grid on;

% 4. Yaw Rate Plot
subplot(2,3,4);
plot(t0, yr, 'LineWidth', 1.5, 'Color', [0.46 0.67 0.18]);
title('Yaw Rate'); xlabel('Time [s]'); ylabel('[deg/s]');
grid on;

% 5. Full GPS Trajectory Map
subplot(2,3,[5 6]);
plot(px, py, 'b', 'LineWidth', 1.5);
title('Complete GPS Trajectory Map'); xlabel('X [m]'); ylabel('Y [m]');
axis equal;
grid on;

sgtitle('Ego Vehicle State Summary', 'FontWeight', 'bold');

%% ---- Figure 2: Sensor Latency Benchmarks ----
% Conversions to milliseconds
gps_lat_ms   = est.gps_pos_report__latency(:, 1) * 1000;
imu_lat_ms   = est.imu_report__latency * 1000; 
lidar_lat_ms = est.lidar_loc_report__latency * 1000;

figure('Name', 'Sensor Latency Profiles', 'Color', 'w');

% 1. GPS Latency
subplot(3,1,1);
plot(t0, gps_lat_ms, 'Color', [0 0.44 0.74]); hold on;
yline(10, 'r--', 'HIGH Threshold', 'LineWidth', 1.2); 
yline(60, 'm--', 'MID Threshold', 'LineWidth', 1.2);
title('GPS Latency Profile'); ylabel('Latency [ms]');
grid on;

% 2. IMU Latency (3 Units stacked)
subplot(3,1,2);
plot(t0, imu_lat_ms, 'LineWidth', 1.1); hold on;
yline(10, 'r--', 'LineWidth', 1.2); 
yline(60, 'm--', 'LineWidth', 1.2);
title('IMU Latency Profiles (3 Units)'); ylabel('Latency [ms]');
legend('IMU 1', 'IMU 2', 'IMU 3', 'Location', 'best');
grid on;

% 3. LiDAR Latency
subplot(3,1,3);
plot(t0, lidar_lat_ms, 'Color', [0.46 0.67 0.18]); hold on;
yline(10, 'r--', 'LineWidth', 1.2); 
yline(60, 'm--', 'LineWidth', 1.2);
title('LiDAR Localization Latency Profile'); xlabel('Time [s]'); ylabel('Latency [ms]');
grid on;

sgtitle('Sensor Subsystem Latency Benchmarks', 'FontWeight', 'bold');