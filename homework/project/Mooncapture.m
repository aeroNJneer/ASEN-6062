%% ASEN-6062 Celestial Mechanics Final Project
%  Nicole James
%  Temporary capture of particles in Earth-Moon CR3BP

clear; clc; close all;

%% Constants
G  = 1.0;
M1 = 5.972e24;       % Earth
M2 = 7.348e22;       % Moon
M3 = 1e-20;          % small asteroid
mu = M2 / (M1 + M2);
pos_earth = [-mu, 0, 0];
pos_moon  = [1 - mu, 0, 0];

%% Lagrange points
L1 = compute_L1(mu);
L2 = compute_L2(mu);
fprintf('L1 Lagrange Point: [%.6f, %.6f, %.6f]\n', L1);
fprintf('Position of Moon relative to L1: [%.6f, %.6f, %.6f]\n', pos_moon - L1);

%% Single trajectory near L1
jacobi_L1 = calc_Jacobi_constant([L1, 0, 0, 0], mu);
fprintf('Jacobi constant at L1: %.6f\n', jacobi_L1);

pos_particle = L1 + [0.001, 0.001, 0];
vel_particle = [0.0, 0.01, 0];
potential = effective_potential(mu, pos_earth, pos_moon, pos_particle);
fprintf('Effective potential at particle position: %.6f\n', potential);

initial_state = [pos_particle, vel_particle];
jacobi_0 = calc_Jacobi_constant(initial_state, mu);
fprintf('Jacobi constant at initial state: %.6f\n', jacobi_0);

t_span = [0, 100];
dt = 0.01;
[t, states] = integrate_orbits(mu, initial_state, t_span, dt);
x  = states(:,1); y  = states(:,2); z  = states(:,3);
vx = states(:,4); vy = states(:,5); vz = states(:,6);

jacobi_constants = arrayfun(@(i) calc_Jacobi_constant(states(i,:), mu), 1:size(states,1));

figure;
subplot(2,1,1);
plot(x, y); hold on;
plot(pos_earth(1), pos_earth(2), 'bo', 'DisplayName', 'Earth');
plot(pos_moon(1), pos_moon(2), 'o', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Moon');
plot(L1(1), L1(2), 'ro', 'DisplayName', 'L1 Point');
xlabel('x'); ylabel('y');
title('Trajectory in the Rotating Frame');
legend; axis equal; hold off;

subplot(2,1,2);
plot(t, jacobi_constants - jacobi_0);
xlabel('Time'); ylabel('Jacobi Constant Deviation');
title('Jacobi Constant Deviation Over Time');

%% Zero-velocity curves (2D)
xg = linspace(-1.5, 1.5, 400);
yg = linspace(-1.5, 1.5, 400);
[Xg, Yg] = meshgrid(xg, yg);
r1 = sqrt((Xg + mu).^2 + Yg.^2);
r2 = sqrt((Xg - 1 + mu).^2 + Yg.^2);
V = (1 - mu) ./ r1 + mu ./ r2 + 0.5 * (Xg.^2 + Yg.^2);

figure;
contour(Xg, Yg, 2*V, [3.10, 3.16, 3.18, 3.20, 3.25]);
hold on;
plot(pos_earth(1), pos_earth(2), 'bo', 'DisplayName', 'Earth');
plot(pos_moon(1), pos_moon(2), 'go', 'DisplayName', 'Moon');
plot(L1(1), L1(2), 'ro', 'DisplayName', 'L1 Point');
xlabel('x'); ylabel('y');
title('Zero-Energy Curves');
legend; axis equal; hold off;

%% 3D Zero-velocity surface
xg3 = linspace(-1.5, 1.5, 100);
yg3 = linspace(-1.5, 1.5, 100);
zg3 = linspace(-0.5, 0.5, 100);
[X3, Y3, Z3] = meshgrid(xg3, yg3, zg3);
r1_3 = sqrt((X3 + mu).^2 + Y3.^2 + Z3.^2);
r2_3 = sqrt((X3 - 1 + mu).^2 + Y3.^2 + Z3.^2);
U3 = (1 - mu) ./ r1_3 + mu ./ r2_3 + 0.5 * (X3.^2 + Y3.^2);
F3 = 2*U3 - jacobi_0;

figure;
isosurface(X3, Y3, Z3, F3, 0);
xlabel('X'); ylabel('Y'); zlabel('Z');
title('3D Zero-Energy Surface');
colormap('default'); lighting gouraud; camlight;

%% Monte Carlo capture search near L1
num_trajectories = 100;
captured_traj = {};
rng('default');

for k = 1:num_trajectories
    pos_p = L1 + (rand(1,3) - 0.5) * 0.2;   % uniform(-0.1, 0.1)
    vel_p = (rand(1,3) - 0.5) * 0.02;        % uniform(-0.01, 0.01)
    state0 = [pos_p, vel_p];
    jacobi = calc_Jacobi_constant(state0, mu);

    [~, sol] = integrate_orbits(mu, state0, [0, 100], 0.01);

    captured_flags = false(size(sol,1), 1);
    for i = 1:size(sol,1)
        captured_flags(i) = is_captured(sol(i,:), mu);
    end

    if any(captured_flags)
        capture_duration = sum(captured_flags) * 0.01;
        fprintf('Captured for %.2f time units, Jacobi=%.4f, state=[%.4f %.4f %.4f %.4f %.4f %.4f]\n', ...
            capture_duration, jacobi, state0);
        captured_traj{end+1} = {state0, jacobi}; %#ok<SAGROW>
    end
end
fprintf('Captured trajectories: %d out of %d\n', length(captured_traj), num_trajectories);

%% Plot a captured trajectory with zero-velocity curves
if ~isempty(captured_traj)
    cap_state = captured_traj{1}{1};
    cap_jacobi = captured_traj{1}{2};
else
    cap_state = [0.90890911, -0.08261452, -0.0432262, -0.00970273, -0.00441134, -0.00462622];
    cap_jacobi = 3.1659;
end

[~, sol_cap] = integrate_orbits(mu, cap_state, [0, 100], 0.01);

figure;
contour(Xg, Yg, 2*V, [3.10, 3.16, 3.18, 3.20, 3.25]);
hold on;
plot(pos_earth(1), pos_earth(2), 'bo', 'DisplayName', 'Earth');
plot(pos_moon(1), pos_moon(2), 'go', 'DisplayName', 'Moon');
plot(L1(1), L1(2), 'ro', 'DisplayName', 'L1 Point');
plot(sol_cap(:,1), sol_cap(:,2), 'DisplayName', 'Trajectory');
xlabel('x'); ylabel('y');
title('Trajectory of Captured Particle with Zero-Energy Curves');
legend; axis equal; hold off;

%% Monte Carlo with Jacobi-based capture check
num_samples = 100;
for k = 1:num_samples
    pos_p = L1 + (rand(1,3) - 0.5) * 0.2;
    vel_p = (rand(1,3) - 0.5) * 0.02;
    state0 = [pos_p, vel_p];

    [t2, sol2] = integrate_orbits(mu, state0, [0, 100], 0.01);
    jc = arrayfun(@(i) calc_Jacobi_constant(sol2(i,:), mu), 1:size(sol2,1));

    if any(jc < jacobi_0)
        fprintf('Trajectory %d is temporarily captured by the Moon.\n', k);
        figure;
        plot(sol2(:,1), sol2(:,2)); hold on;
        plot(pos_earth(1), pos_earth(2), 'bo', 'DisplayName', 'Earth');
        plot(pos_moon(1), pos_moon(2), 'o', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Moon');
        plot(L1(1), L1(2), 'ro', 'DisplayName', 'L1 Point');
        xlabel('x'); ylabel('y');
        title(sprintf('Trajectory %d - Temporarily Captured', k));
        legend; hold off;
    end
end

%% ==================== Local Functions ====================

function L1 = compute_L1(mu)
    L1 = [1 - (mu/3)^(1/3), 0, 0];
end

function L2 = compute_L2(mu)
    L2 = [1 + (mu/3)^(1/3), 0, 0];
end

function U = effective_potential(mu, pos_earth, pos_moon, pos_particle)
    r1 = norm(pos_particle - pos_earth);
    r2 = norm(pos_particle - pos_moon);
    U = (1 - mu)/r1 + mu/r2 + 0.5 * norm(pos_particle(1:2))^2;
end

function J = calc_Jacobi_constant(state, mu)
    pos_earth = [-mu, 0, 0];
    pos_moon  = [1 - mu, 0, 0];
    U = effective_potential(mu, pos_earth, pos_moon, state(1:3));
    v_sq = norm(state(4:6))^2;
    J = 2*U - v_sq;
end

function dstate = equations_of_motion(~, state, mu)
    x = state(1); y = state(2); z = state(3);
    vx = state(4); vy = state(5); vz = state(6);

    r1 = sqrt((x + mu)^2 + y^2 + z^2);
    r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2);

    dstate = zeros(6,1);
    dstate(1) = vx;
    dstate(2) = vy;
    dstate(3) = vz;
    dstate(4) = 2*vy + x - (1 - mu)*(x + mu)/r1^3 - mu*(x - 1 + mu)/r2^3;
    dstate(5) = -2*vx + y - (1 - mu)*y/r1^3 - mu*y/r2^3;
    dstate(6) = -(1 - mu)*z/r1^3 - mu*z/r2^3;
end

function [t, states] = integrate_orbits(mu, initial_state, t_span, dt)
    t_eval = t_span(1):dt:t_span(2);
    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    [t, states] = ode89(@(t, s) equations_of_motion(t, s, mu), t_eval, initial_state, opts);
end

function flag = is_captured(state, mu)
    x = state(1); y = state(2); z = state(3);
    vx = state(4); vy = state(5); vz = state(6);
    x_moon = 1 - mu;

    r2 = sqrt((x - x_moon)^2 + y^2 + z^2);
    r_hill = (mu / 3)^(1/3);

    v_rel_sq = vx^2 + vy^2 + vz^2;
    E2 = 0.5 * v_rel_sq - mu / r2;

    flag = (r2 < r_hill) && (E2 < 0);
end
