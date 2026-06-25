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
spd  = sqrt(vx.^2 + vy.^2); % Calculated speed for color map
acc  = sqrt(ax.^2 + ay.^2);
hdg  = rad2deg(heading);
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

% 5. Full GPS Trajectory Map (With Color-Coded Speed)
subplot(2,3,[5 6]);

% Create a patch object using the coordinates. Z values are set to 0.
% The 'CData' property defines the color variation using your speed array.
h_patch = patch([px; NaN], [py; NaN], [spd; NaN], [spd; NaN], ...
                'EdgeColor', 'interp', ...
                'FaceColor', 'none', ...
                'LineWidth', 2.0);

title('Complete GPS Trajectory Map'); xlabel('X [m]'); ylabel('Y [m]');
axis equal;
grid on;

% Add and label the color intensity bar
cb = colorbar;
ylabel(cb, 'Speed [m/s]', 'FontSize', 9);
colormap(jet); % Uses standard 'jet' spectrum (Blue=Slow, Red=Fast)

sgtitle('Ego Vehicle State Summary', 'FontWeight', 'bold');