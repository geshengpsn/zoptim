//! By convention, root.zig is the root source file when making a package.
//!
const std = @import("std");
const Io = std.Io;
const zla = @import("zla");
const zplotly = @import("zplotly");

const P = zla.Mat(f64, 2, 2).init(.{
    1.0, 0.0,
    0.0, 2.0,
});

const Q: @Vector(2, f64) = .{ 1.0, 0.0 };

fn f(x: @Vector(2, f64)) f64 {
    var temp: @Vector(2, f64) = undefined;
    P.vec_mul(&x, &temp);
    temp = temp * @as(@Vector(2, f64), @splat(0.5));
    return @reduce(.Add, x * (temp + Q));
}

fn df(x: @Vector(2, f64)) @Vector(2, f64) {
    var temp: @Vector(2, f64) = undefined;
    P.vec_mul(&x, &temp);
    return temp + Q;
}

fn hessian(x: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
    _ = x;
    return P;
}

test "aa" {
    const result = f(.{ 1.0, 2.0 });
    try std.testing.expectEqual(result, 5.5);
}

test "df" {
    const result = df(.{ 1.0, 2.0 });
    try std.testing.expectEqual(result, .{ 2.0, 4.0 });
}

test "hessian" {
    const result = hessian(.{ 1.0, 2.0 });
    try std.testing.expectEqual(result, zla.Mat(f64, 2, 2).init(.{
        1.0, 0.0,
        0.0, 2.0,
    }));
}

test "GD no line search" {
    var x: @Vector(2, f64) = .{ 1.0, 2.0 };
    var count: usize = 0;
    const learning_rate: @Vector(2, f64) = @splat(0.1);
    while (true) {
        const dir = -df(x);
        if (@sqrt(@reduce(.Add, dir * dir)) < 1e-9) break;
        x = x + learning_rate * dir;
        // std.debug.print("{} dir: {any} x: {any} error: {any}\n", .{ count, dir, x, @sqrt(@reduce(.Add, dir * dir)) });
        count += 1;
    }
    try std.testing.expectEqual(count, 204);
}

fn armijo(comptime n: usize, comptime T: type, func: fn (@Vector(n, T)) T, x: @Vector(n, T), grad: @Vector(n, T), dir: @Vector(n, T)) T {
    var t: T = 1.0;
    const f_x = func(x);

    while (func(x + @as(@Vector(n, T), @splat(t)) * dir) > f_x + 0.4 * t * @reduce(.Add, grad * dir)) {
        t *= 0.5;
    }
    return t;
}

test "GD backtracking Armijo" {
    var x: @Vector(2, f64) = .{ 1.0, 2.0 };
    var count: usize = 0;
    while (true) {
        const grad = df(x);
        if (@sqrt(@reduce(.Add, grad * grad)) < 1e-9) break;
        const dir = -grad;
        const t = armijo(2, f64, f, x, grad, dir);
        x = x + @as(@Vector(2, f64), @splat(t)) * dir;
        count += 1;
    }
    try std.testing.expectEqual(count, 2);
}

const funcs = struct {
    fn f(x: @Vector(2, f64)) f64 {
        return @exp(x[0] + 3.0 * x[1] - 0.1) + @exp(x[0] - 3.0 * x[1] - 0.1) + @exp(-x[0] - 0.1);
    }

    fn grad(x: @Vector(2, f64)) @Vector(2, f64) {
        const a = @exp(x[0] + 3.0 * x[1] - 0.1);
        const b = @exp(x[0] - 3.0 * x[1] - 0.1);
        const c = @exp(-x[0] - 0.1);

        return .{ a + b - c, 3.0 * (a - b) };
    }

    fn hess(x: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
        const a = @exp(x[0] + 3.0 * x[1] - 0.1);
        const b = @exp(x[0] - 3.0 * x[1] - 0.1);
        const c = @exp(-x[0] - 0.1);

        return zla.Mat(f64, 2, 2).init(.{
            a + b + c,     3.0 * (a - b),
            3.0 * (a - b), 9.0 * (a + b),
        });
    }
};

const plot_grid_size = 81;
const plot_max_iters = 64;

const OptimizationPath = struct {
    x: [plot_max_iters + 1]f64,
    y: [plot_max_iters + 1]f64,
    z: [plot_max_iters + 1]f64,
    len: usize,
};

pub fn plotNewtonArmijo(allocator: std.mem.Allocator, io: Io) !void {
    const x_values = linspace(plot_grid_size, -1.5, 1.5);
    const y_values = linspace(plot_grid_size, -2.2, 2.2);
    const z_values = objectiveGrid(plot_grid_size, &x_values, &y_values);
    const path = try collectNewtonArmijoPath();

    const path_x = path.x[0..path.len];
    const path_y = path.y[0..path.len];
    const path_z = path.z[0..path.len];

    const surface_data = .{
        .{
            .x = x_values[0..],
            .y = y_values[0..],
            .z = z_values[0..],
            .type = "surface",
            .name = "f(x)",
            .colorscale = "Viridis",
            .opacity = 0.82,
        },
        .{
            .x = path_x,
            .y = path_y,
            .z = path_z,
            .type = "scatter3d",
            .mode = "lines+markers",
            .name = "Newton + Armijo",
            .line = .{ .color = "#ff2d55", .width = 7 },
            .marker = .{ .color = "#ffffff", .size = 4 },
        },
    };
    const surface_layout = .{
        .title = .{ .text = "Objective Surface with Newton + Armijo Iterates" },
        .scene = .{
            .xaxis = .{ .title = .{ .text = "x0" } },
            .yaxis = .{ .title = .{ .text = "x1" } },
            .zaxis = .{ .title = .{ .text = "f(x)" } },
        },
    };
    try zplotly.plot(allocator, io, "zoptim_newton_armijo_surface", surface_data, surface_layout);
}

fn linspace(comptime n: usize, min: f64, max: f64) [n]f64 {
    comptime {
        if (n < 2) @compileError("linspace requires at least two points");
    }

    var values: [n]f64 = undefined;
    const step = (max - min) / @as(f64, @floatFromInt(n - 1));
    for (&values, 0..) |*value, i| {
        value.* = min + step * @as(f64, @floatFromInt(i));
    }
    return values;
}

fn objectiveGrid(comptime n: usize, x_values: *const [n]f64, y_values: *const [n]f64) [n][n]f64 {
    var z_values: [n][n]f64 = undefined;
    for (0..n) |row| {
        for (0..n) |col| {
            z_values[row][col] = funcs.f(.{ x_values.*[col], y_values.*[row] });
        }
    }
    return z_values;
}

fn collectNewtonArmijoPath() !OptimizationPath {
    var path: OptimizationPath = undefined;
    path.len = 0;

    var x: @Vector(2, f64) = .{ 1.0, 2.0 };
    appendPathPoint(&path, x);

    var iter: usize = 0;
    while (true) {
        const hess = funcs.hess(x);
        var hess_inv: @TypeOf(hess) = undefined;
        try hess.mat_inv(&hess_inv);

        const grad = funcs.grad(x);

        var dir: @TypeOf(grad) = undefined;
        hess_inv.vec_mul(&grad, &dir);
        dir = -dir;

        const err: f64 = -@reduce(.Add, grad * dir) / 2.0;
        if (err < 1e-9) break;

        if (iter == plot_max_iters) return error.TooManyIterations;

        const t = armijo(2, f64, funcs.f, x, grad, dir);
        x = x + @as(@Vector(2, f64), @splat(t)) * dir;
        appendPathPoint(&path, x);

        iter += 1;
    }

    return path;
}

fn appendPathPoint(path: *OptimizationPath, x: @Vector(2, f64)) void {
    path.x[path.len] = x[0];
    path.y[path.len] = x[1];
    path.z[path.len] = funcs.f(x);
    path.len += 1;
}

test "newton no line search" {
    var x: @Vector(2, f64) = .{ 1.0, 2.0 };
    var count: usize = 0;
    while (true) {
        const hess = funcs.hess(x);
        var hess_inv: @TypeOf(hess) = undefined;
        hess.mat_inv(&hess_inv) catch unreachable;

        const grad = funcs.grad(x);

        var dir: @TypeOf(grad) = undefined;
        hess_inv.vec_mul(&grad, &dir);
        dir = -dir;

        const err: f64 = -@reduce(.Add, grad * dir) / 2;
        if (err < 1e-9) break;

        x = x + dir;

        count += 1;
    }
    try std.testing.expectEqual(count, 10);
}

test "newton armijo" {
    var x: @Vector(2, f64) = .{ 2.0, 4.0 };
    var count: usize = 0;
    while (true) {
        const hess = funcs.hess(x);
        var hess_inv: @TypeOf(hess) = undefined;
        hess.mat_inv(&hess_inv) catch unreachable;

        const grad = funcs.grad(x);

        var dir: @TypeOf(grad) = undefined;
        hess_inv.vec_mul(&grad, &dir);
        dir = -dir;
        const t = armijo(2, f64, funcs.f, x, grad, dir);

        const err: f64 = -@reduce(.Add, grad * dir) / 2;
        if (err < 1e-9) break;

        x = x + @as(@Vector(2, f64), @splat(t)) * dir;

        count += 1;
    }
    try std.testing.expectEqual(count, 17);
}

fn EqConstraints(comptime n: usize, comptime p: usize, comptime T: type) type {
    return struct {
        a: zla.Mat(T, p, n),
        b: @Vector(p, T),
    };
}

fn InequalityConstraints(comptime n: usize, comptime m: usize, comptime T: type) type {
    return struct {
        h: fn (@Vector(n, T)) @Vector(m, T),
        jac: fn (@Vector(n, T)) zla.Mat(T, m, n),
    };
}

fn ProblemNoConstraints(comptime n: usize, comptime T: type) type {
    return struct {
        func: fn (@Vector(n, T)) T,
        grad: fn (@Vector(n, T)) @Vector(n, T),
        hess: fn (@Vector(n, T)) zla.Mat(T, n, n),
    };
}

fn Solver(comptime n: usize, comptime T: type) type {
    return struct {
        solve: fn (*const zla.Mat(T, n, n), *const @Vector(n, T), *@Vector(n, T)) void,
    };
}

fn optimizeNoConstraints(
    comptime n: usize,
    comptime T: type,
    objective: type,
    solver: type,
    init_x: ?@Vector(n, T),
    e: T,
) @Vector(n, T) {
    var x: @Vector(n, T) = init_x orelse @as(@Vector(n, T), @splat(0));
    while (true) {
        const hess = objective.hess(x);
        const grad = objective.grad(x);
        var dir: @TypeOf(grad) = undefined;
        solver.solve(&hess, &grad, &dir);
        const t = armijo(n, T, objective.f, x, grad, dir);
        const err: T = -@reduce(.Add, grad * dir) / @as(T, 2.0);
        if (err < e) break;
        x = x + @as(@Vector(n, T), @splat(t)) * dir;
    }
    return x;
}

test "optimizeNoConstraints" {
    const init_x: @Vector(2, f64) = .{ 2.0, 4.0 };
    const result = optimizeNoConstraints(2, f64, funcs, struct {
        fn solve(a: *const zla.Mat(f64, 2, 2), b: *const @Vector(2, f64), x: *@Vector(2, f64)) void {
            a.solve_lu(-b, x) catch unreachable;
        }
    }, init_x, 1e-10);
    try std.testing.expectApproxEqAbs(@reduce(.Add, funcs.grad(result)), 0.0, 1e-4);
}

fn optimality_conditions_armijo(
    comptime n: usize,
    comptime p: usize,
    comptime T: type,
    r_norm: fn (@Vector(n, T), @Vector(p, T)) T,
    x: @Vector(n, T),
    dual: @Vector(n, T),
    x_dir: @Vector(n, T),
    dual_dir: @Vector(p, T),
    alpha: T,
    beta: T,
) T {
    std.debug.assert(0.0 < alpha and alpha < 0.5);
    std.debug.assert(0.0 < beta and beta < 1.0);

    var t: T = 1.0;
    const norm = r_norm(x, dual);

    while (r_norm(
        x + @as(@Vector(n, T), @splat(t)) * x_dir,
        dual + @as(@Vector(n, T), @splat(t)) * dual_dir,
    ) > (1 - alpha * t) * norm) {
        t *= beta;
    }
    return t;
}

fn Params(comptime T: type) type {
    return struct {
        e: T = 1e-9,
        alpha: T = 0.4,
        beta: T = 0.5,
    };
}

fn optimizeEqConstraints(
    comptime n: usize,
    comptime p: usize,
    comptime T: type,
    objective: type,
    eq_constraints: EqConstraints(p, T),
    solver: type,
    init_x: ?@Vector(n, T),
    params: ?Params(T),
) void {
    const param: Params(T) = params orelse .{};
    var x: @Vector(n, T) = init_x orelse @as(@Vector(n, T), @splat(0));
    var dual: @Vector(p, T) = @as(@Vector(p, T), @splat(0));
    std.debug.assert(param.e > 0.0);
    std.debug.assert(0.0 < param.alpha and param.alpha < 0.5);
    std.debug.assert(0.0 < param.beta and param.beta < 1.0);
    std.debug.assert(!std.math.isInf(objective.f(x)) and !std.math.isNan(objective.f(x)));

    while (true) {
        const hess = objective.hess(x);
        const grad = objective.grad(x);
        var x_dir: @TypeOf(grad) = undefined;
        var dual_dir: @TypeOf(grad) = undefined;
        const r_dual = grad + eq_constraints.a.transpose();
        solver.solve(&hess, &grad, &dir);
        var t: T = 1.0;
        const norm = r_norm(x, dual);

        while (r_norm(
            x + @as(@Vector(n, T), @splat(t)) * x_dir,
            dual + @as(@Vector(n, T), @splat(t)) * dual_dir,
        ) > (1 - alpha * t) * norm) {
            t *= beta;
        }
        const err: T = -@reduce(.Add, grad * dir) / @as(T, 2.0);
        if (err < p.e) break;
        x = x + @as(@Vector(n, T), @splat(t)) * dir;
    }
    return x;
}
