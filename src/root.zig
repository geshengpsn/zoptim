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

fn f_qp(x: @Vector(2, f64)) f64 {
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
    const result = f_qp(.{ 1.0, 2.0 });
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
        const t = armijo(2, f64, f_qp, x, grad, dir);
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
    f: fn (x: @Vector(n, T)) T,
    grad: fn (x: @Vector(n, T)) @Vector(n, T),
    hess: fn (x: @Vector(n, T)) zla.Mat(T, n, n),
    solver: fn (hess: *const zla.Mat(T, n, n), grad: *const @Vector(n, T), dir: *@Vector(n, T)) void,
    init_x: ?@Vector(n, T),
    e: T,
) @Vector(n, T) {
    var x: @Vector(n, T) = init_x orelse @as(@Vector(n, T), @splat(0));
    while (true) {
        const hess_val = hess(x);
        const grad_val = grad(x);
        var dir: @TypeOf(grad_val) = undefined;
        solver(&hess_val, &grad_val, &dir);
        const t = armijo(n, T, f, x, grad_val, dir);
        const err: T = -@reduce(.Add, grad_val * dir) / @as(T, 2.0);
        if (err < e) break;
        x = x + @as(@Vector(n, T), @splat(t)) * dir;
    }
    return x;
}

test "optimizeNoConstraints" {
    const init_x: @Vector(2, f64) = .{ 2.0, 4.0 };
    const result = optimizeNoConstraints(2, f64, funcs.f, funcs.grad, funcs.hess, struct {
        fn solve(a: *const zla.Mat(f64, 2, 2), b: *const @Vector(2, f64), x: *@Vector(2, f64)) void {
            const rhs = -b.*;
            a.solve_cholesky(&rhs, x) catch unreachable;
        }
    }.solve, init_x, 1e-10);
    try std.testing.expectApproxEqAbs(@reduce(.Add, funcs.grad(result)), 0.0, 1e-4);
}

// fn optimality_conditions_armijo(
//     comptime n: usize,
//     comptime p: usize,
//     comptime T: type,
//     r_norm: fn (@Vector(n, T), @Vector(p, T)) T,
//     x: @Vector(n, T),
//     dual: @Vector(n, T),
//     x_dir: @Vector(n, T),
//     dual_dir: @Vector(p, T),
//     alpha: T,
//     beta: T,
// ) T {
//     std.debug.assert(0.0 < alpha and alpha < 0.5);
//     std.debug.assert(0.0 < beta and beta < 1.0);

//     var t: T = 1.0;
//     const norm = r_norm(x, dual);

//     while (r_norm(
//         x + @as(@Vector(n, T), @splat(t)) * x_dir,
//         dual + @as(@Vector(n, T), @splat(t)) * dual_dir,
//     ) > (1 - alpha * t) * norm) {
//         t *= beta;
//     }
//     return t;
// }

fn Params(comptime T: type) type {
    return struct {
        e: T = 1e-9,
        alpha: T = 0.4,
        beta: T = 0.5,
    };
}

fn r(
    comptime n: usize,
    comptime p: usize,
    comptime T: type,
    grad_val: *const @Vector(n, T),
    a: *const zla.Mat(T, p, n),
    b: *const @Vector(p, T),
    x: *const @Vector(n, T),
    dual: *const @Vector(p, T),
    r_dual: *@Vector(n, T),
    r_prim: *@Vector(p, T),
) void {
    const a_transpose = a.transpose();
    var a_transpose_dual: @Vector(n, T) = undefined;
    a_transpose.vec_mul(dual, &a_transpose_dual);

    r_dual.* = grad_val.* + a_transpose_dual;
    var ax: @Vector(p, T) = undefined;
    a.vec_mul(x, &ax);
    r_prim.* = ax - b.*;
}

fn r_norm(
    comptime n: usize,
    comptime p: usize,
    comptime T: type,
    r_dual: *const @Vector(n, T),
    r_prim: *const @Vector(p, T),
) T {
    return @sqrt(@reduce(.Add, r_dual.* * r_dual.*) + @reduce(.Add, r_prim.* * r_prim.*));
}

fn EqConstraintsResult(comptime n: usize, comptime p: usize, comptime T: type) type {
    return struct {
        x: @Vector(n, T),
        dual: @Vector(p, T),
    };
}

fn optimizeEqConstraints(
    comptime n: usize,
    comptime p: usize,
    comptime T: type,
    f: fn (x: @Vector(n, T)) T,
    grad: fn (x: @Vector(n, T)) @Vector(n, T),
    hess: fn (x: @Vector(n, T)) zla.Mat(T, n, n),
    eq_constraints: EqConstraints(n, p, T),
    solver: fn (
        hess: *const zla.Mat(T, n, n),
        a: *const zla.Mat(T, p, n),
        r_dual: *const @Vector(n, T),
        r_prim: *const @Vector(p, T),
        x_step: *@Vector(n, T),
        dual_step: *@Vector(p, T),
    ) void,
    init_x: @Vector(n, T),
    params: ?Params(T),
) EqConstraintsResult(n, p, T) {
    const param: Params(T) = params orelse .{};
    var x: @Vector(n, T) = init_x;
    var dual: @Vector(p, T) = @as(@Vector(p, T), @splat(0));
    std.debug.assert(param.e > 0.0);
    std.debug.assert(0.0 < param.alpha and param.alpha < 0.5);
    std.debug.assert(0.0 < param.beta and param.beta < 1.0);
    std.debug.assert(!std.math.isInf(f(x)) and !std.math.isNan(f(x)));
    var r_dual: @Vector(n, T) = undefined;
    var r_prim: @Vector(p, T) = undefined;
    var x_step: @Vector(n, T) = undefined;
    var dual_step: @Vector(p, T) = undefined;
    while (true) {
        const hess_val = hess(x);
        const grad_val = grad(x);

        r(n, p, T, &grad_val, &eq_constraints.a, &eq_constraints.b, &x, &dual, &r_dual, &r_prim);
        const norm = r_norm(n, p, T, &r_dual, &r_prim);
        if (norm < param.e) break;

        solver(&hess_val, &eq_constraints.a, &r_dual, &r_prim, &x_step, &dual_step);

        var t: T = 1.0;
        while (true) {
            const x_t = x + @as(@Vector(n, T), @splat(t)) * x_step;
            const dual_t = dual + @as(@Vector(p, T), @splat(t)) * dual_step;
            const grad_t = grad(x_t);
            var r_dual_t: @Vector(n, T) = undefined;
            var r_prim_t: @Vector(p, T) = undefined;
            r(
                n,
                p,
                T,
                &grad_t,
                &eq_constraints.a,
                &eq_constraints.b,
                &x_t,
                &dual_t,
                &r_dual_t,
                &r_prim_t,
            );
            const norm_t = r_norm(n, p, T, &r_dual_t, &r_prim_t);
            if (norm_t <= (1 - param.alpha * t) * norm) break;
            t *= param.beta;
        }

        x = x + @as(@Vector(n, T), @splat(t)) * x_step;
        dual = dual + @as(@Vector(p, T), @splat(t)) * dual_step;
    }
    return .{ .x = x, .dual = dual };
}

test "optimizeEqConstraints QP" {
    const eq_constraints = EqConstraints(2, 1, f64){
        .a = zla.Mat(f64, 1, 2).init(.{ 1.0, 1.0 }),
        .b = .{1.0},
    };
    const result = optimizeEqConstraints(2, 1, f64, f_qp, df, hessian, eq_constraints, struct {
        fn solve(
            hess: *const zla.Mat(f64, 2, 2),
            a: *const zla.Mat(f64, 1, 2),
            r_dual: *const @Vector(2, f64),
            r_prim: *const @Vector(1, f64),
            x_step: *@Vector(2, f64),
            dual_step: *@Vector(1, f64),
        ) void {
            const a_transpose = a.transpose();
            var kkt = zla.Mat(f64, 3, 3).init(.{
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
            });
            kkt.set_block(0, 0, hess);
            kkt.set_block(0, 2, &a_transpose);
            kkt.set_block(2, 0, a);

            const rhs: @Vector(3, f64) = .{ -r_dual.*[0], -r_dual.*[1], -r_prim.*[0] };
            var step: @Vector(3, f64) = undefined;
            kkt.solve_lu(&rhs, &step) catch unreachable;

            x_step.* = .{ step[0], step[1] };
            dual_step.* = .{step[2]};
        }
    }.solve, .{ 0.0, 0.0 }, .{ .e = 1e-10 });

    try std.testing.expectApproxEqAbs(1.0 / 3.0, result.x[0], 1e-9);
    try std.testing.expectApproxEqAbs(2.0 / 3.0, result.x[1], 1e-9);
    try std.testing.expectApproxEqAbs(1.0, result.x[0] + result.x[1], 1e-9);
    try std.testing.expectApproxEqAbs(-4.0 / 3.0, result.dual[0], 1e-9);
}

fn InequalityConstraints(comptime n: usize, comptime m: usize, comptime T: type) type {
    return struct {
        f: fn (x: @Vector(n, T)) @Vector(m, T),
        jac: fn (x: @Vector(n, T)) zla.Mat(T, m, n),
        hess_sum: fn (x: @Vector(n, T), ieq_dual: @Vector(m, T)) zla.Mat(T, n, n),
    };
}

fn h_jac_diag_ieq_dual(
    comptime n: usize,
    comptime m: usize,
    comptime T: type,
    h_jac: *const zla.Mat(T, m, n),
    ieq_dual: *const @Vector(m, T),
) zla.Mat(T, m, n) {
    var h_jac_diag: zla.Mat(T, m, n) = undefined;
    for (0..n) |i| {
        h_jac_diag.set_col(i, -h_jac.get_col(i) * ieq_dual.*);
    }
    return h_jac_diag;
}

test "h_jac_diag_cent" {
    const h_jac = zla.Mat(f64, 2, 2).init(.{
        1.0, 2.0,
        3.0, 4.0,
    });
    const ieq_dual: @Vector(2, f64) = .{ 1.0, 2.0 };
    const result = h_jac_diag_ieq_dual(2, 2, f64, &h_jac, &ieq_dual);
    try std.testing.expectEqual(result, zla.Mat(f64, 2, 2).init(.{
        -1.0, -2.0,
        -6.0, -8.0,
    }));
}

fn ie_r(
    comptime n: usize,
    comptime m: usize,
    comptime p: usize,
    comptime T: type,
    x: *const @Vector(n, T),
    ieq_dual: *const @Vector(m, T),
    eq_dual: *const @Vector(p, T),
    x_grad: *const @Vector(n, T),
    h: *const @Vector(m, T),
    slack: *const @Vector(m, T),
    h_jac_transpose: *const zla.Mat(T, n, m),
    a: *const zla.Mat(T, p, n),
    b: *const @Vector(p, T),
    t: T,
    r_dual: *@Vector(n, T),
    r_cent: *@Vector(m, T),
    r_prim: *@Vector(p, T),
    r_ieq: *@Vector(m, T),
) void {
    var a_transpose_dual: @Vector(n, T) = undefined;
    a.transpose().vec_mul(eq_dual, &a_transpose_dual);
    var h_jac_transpose_cent: @Vector(n, T) = undefined;
    h_jac_transpose.vec_mul(ieq_dual, &h_jac_transpose_cent);
    r_dual.* = x_grad.* + a_transpose_dual + h_jac_transpose_cent;
    r_cent.* = slack.* * ieq_dual.* - @as(@Vector(m, T), @splat(1.0 / t));
    var ax: @Vector(p, T) = undefined;
    a.vec_mul(x, &ax);
    r_prim.* = ax - b.*;
    r_ieq.* = h.* + slack.*;
}

fn ie_r_norm(
    comptime n: usize,
    comptime m: usize,
    comptime p: usize,
    comptime T: type,
    r_dual: *const @Vector(n, T),
    r_cent: *const @Vector(m, T),
    r_prim: *const @Vector(p, T),
    r_ieq: *const @Vector(m, T),
) T {
    return @sqrt(@reduce(.Add, r_prim.* * r_prim.*) + @reduce(.Add, r_dual.* * r_dual.*) + @reduce(.Add, r_cent.* * r_cent.*) + @reduce(.Add, r_ieq.* * r_ieq.*));
}

const GeneralConstraintsInitMode = enum {
    feasible,
    infeasible,
};

fn GeneralConstraintsParams(comptime T: type) type {
    return struct {
        e_feasible: T = 1e-6,
        e_gap: T = 1e-6,
        miu: T = 10.0,
        alpha: T = 0.1,
        beta: T = 0.5,
    };
}

fn GeneralConstraintsResult(comptime n: usize, comptime m: usize, comptime p: usize, comptime T: type) type {
    return struct {
        x: @Vector(n, T),
        ieq_dual: @Vector(m, T),
        eq_dual: @Vector(p, T),
    };
}

fn positiveInitialSlack(comptime m: usize, comptime T: type, h: @Vector(m, T)) @Vector(m, T) {
    const h_arr = @as([m]T, h);
    var slack_arr: [m]T = undefined;
    for (0..m) |i| {
        slack_arr[i] = if (h_arr[i] < 0.0) -h_arr[i] else 1.0;
    }
    return @as(@Vector(m, T), slack_arr);
}

fn optimizeGeneralConstraints(
    comptime n: usize,
    comptime m: usize,
    comptime p: usize,
    comptime T: type,
    comptime init_mode: GeneralConstraintsInitMode,
    f: fn (x: @Vector(n, T)) T,
    grad: fn (x: @Vector(n, T)) @Vector(n, T),
    hess: fn (x: @Vector(n, T)) zla.Mat(T, n, n),
    eq_constraints: EqConstraints(n, p, T),
    ieq_constraints: InequalityConstraints(n, m, T),
    solver: fn (
        block_hess: *const zla.Mat(T, n, n),
        block_h_jac_transpose: *const zla.Mat(T, n, m),
        block_a: *const zla.Mat(T, p, n),
        block_h_jac_diag_iedual: *const zla.Mat(T, m, n),
        block_diag_h: *const @Vector(m, T),
        r_dual: *const @Vector(n, T),
        r_cent: *const @Vector(m, T),
        r_prim: *const @Vector(p, T),
        x_step: *@Vector(n, T),
        ieq_dual_step: *@Vector(m, T),
        eq_dual_step: *@Vector(p, T),
    ) void,
    init_x: @Vector(n, T),
    params: ?GeneralConstraintsParams(T),
) GeneralConstraintsResult(n, m, p, T) {
    const param: GeneralConstraintsParams(T) = params orelse .{};
    var x: @Vector(n, T) = init_x;
    const init_h = ieq_constraints.f(x);
    var slack: @Vector(m, T) = switch (init_mode) {
        .feasible => -init_h,
        .infeasible => positiveInitialSlack(m, T, init_h),
    };
    var ieq_dual: @Vector(m, T) = @as(@Vector(m, T), @splat(1));
    var eq_dual: @Vector(p, T) = @as(@Vector(p, T), @splat(0));

    std.debug.assert(param.e_feasible > 0.0);
    std.debug.assert(param.e_gap > 0.0);
    std.debug.assert(param.miu > 1.0);
    std.debug.assert(0.0 < param.alpha and param.alpha < 0.5);
    std.debug.assert(0.0 < param.beta and param.beta < 1.0);
    std.debug.assert(!std.math.isInf(f(x)) and !std.math.isNan(f(x)));

    var r_dual: @Vector(n, T) = undefined;
    var r_cent: @Vector(m, T) = undefined;
    var r_prim: @Vector(p, T) = undefined;
    var r_ieq: @Vector(m, T) = undefined;

    var x_step: @Vector(n, T) = undefined;
    var ieq_dual_step: @Vector(m, T) = undefined;
    var eq_dual_step: @Vector(p, T) = undefined;
    var slack_step: @Vector(m, T) = undefined;

    while (true) {
        const grad_val = grad(x);
        const hess_val = hess(x);
        const h_val = ieq_constraints.f(x);
        if (init_mode == .feasible) {
            slack = -h_val;
        }
        const h_jac_val = ieq_constraints.jac(x);
        const hess_sum_val = ieq_constraints.hess_sum(x, ieq_dual);
        const block_h_jac_transpose = h_jac_val.transpose();
        const gap = @reduce(.Add, slack * ieq_dual);
        std.debug.assert(gap > 0.0);
        const h_arr = @as([m]T, h_val);
        const slack_arr = @as([m]T, slack);
        const ieq_dual_arr = @as([m]T, ieq_dual);
        for (0..m) |i| {
            if (init_mode == .feasible) {
                std.debug.assert(h_arr[i] < 0.0);
            }
            std.debug.assert(slack_arr[i] > 0.0);
            std.debug.assert(ieq_dual_arr[i] > 0.0);
        }
        const centering_t = param.miu * @as(T, @floatFromInt(m)) / gap;
        ie_r(
            n,
            m,
            p,
            T,
            &x,
            &ieq_dual,
            &eq_dual,
            &grad_val,
            &h_val,
            &slack,
            &block_h_jac_transpose,
            &eq_constraints.a,
            &eq_constraints.b,
            centering_t,
            &r_dual,
            &r_cent,
            &r_prim,
            &r_ieq,
        );
        const norm = ie_r_norm(n, m, p, T, &r_dual, &r_cent, &r_prim, &r_ieq);
        if (@reduce(.Add, r_prim * r_prim) < param.e_feasible * param.e_feasible and @reduce(.Add, r_ieq * r_ieq) < param.e_feasible * param.e_feasible and @reduce(.Add, r_dual * r_dual) < param.e_feasible * param.e_feasible and gap < param.e_gap) break;
        var block_hess: zla.Mat(T, n, n) = undefined;
        hess_val.mat_add(&hess_sum_val, &block_hess);
        const block_h_jac_diag_ieq_dual = h_jac_diag_ieq_dual(n, m, T, &h_jac_val, &ieq_dual);
        const solver_r_cent = r_cent - ieq_dual * r_ieq;
        solver(
            &block_hess,
            &block_h_jac_transpose,
            &eq_constraints.a,
            &block_h_jac_diag_ieq_dual,
            &slack,
            &r_dual,
            &solver_r_cent,
            &r_prim,
            &x_step,
            &ieq_dual_step,
            &eq_dual_step,
        );
        h_jac_val.vec_mul(&x_step, &slack_step);
        slack_step = -r_ieq - slack_step;

        var t: T = 1.0;
        const ieq_dual_step_arr = @as([m]T, ieq_dual_step);
        const slack_step_arr = @as([m]T, slack_step);
        for (0..m) |i| {
            if (ieq_dual_step_arr[i] < 0.0) {
                t = @min(t, -0.99 * ieq_dual_arr[i] / ieq_dual_step_arr[i]);
            }
            if (slack_step_arr[i] < 0.0) {
                t = @min(t, -0.99 * slack_arr[i] / slack_step_arr[i]);
            }
        }
        backtracking: while (true) {
            var r_dual_t: @Vector(n, T) = undefined;
            var r_cent_t: @Vector(m, T) = undefined;
            var r_prim_t: @Vector(p, T) = undefined;
            var r_ieq_t: @Vector(m, T) = undefined;
            const x_t = x + @as(@Vector(n, T), @splat(t)) * x_step;
            const slack_t = switch (init_mode) {
                .feasible => -ieq_constraints.f(x_t),
                .infeasible => slack + @as(@Vector(m, T), @splat(t)) * slack_step,
            };
            const ieq_dual_t = ieq_dual + @as(@Vector(m, T), @splat(t)) * ieq_dual_step;
            const eq_dual_t = eq_dual + @as(@Vector(p, T), @splat(t)) * eq_dual_step;
            const grad_val_t = grad(x_t);
            const h_val_t = ieq_constraints.f(x_t);
            const h_val_t_arr = @as([m]T, h_val_t);
            const slack_t_arr = @as([m]T, slack_t);
            const ieq_dual_t_arr = @as([m]T, ieq_dual_t);
            for (0..m) |i| {
                if ((init_mode == .feasible and h_val_t_arr[i] >= 0.0) or slack_t_arr[i] <= 0.0 or ieq_dual_t_arr[i] <= 0.0) {
                    t *= param.beta;
                    continue :backtracking;
                }
            }
            const block_h_jac_transpose_t = ieq_constraints.jac(x_t).transpose();
            ie_r(
                n,
                m,
                p,
                T,
                &x_t,
                &ieq_dual_t,
                &eq_dual_t,
                &grad_val_t,
                &h_val_t,
                &slack_t,
                &block_h_jac_transpose_t,
                &eq_constraints.a,
                &eq_constraints.b,
                centering_t,
                &r_dual_t,
                &r_cent_t,
                &r_prim_t,
                &r_ieq_t,
            );
            const norm_t = ie_r_norm(n, m, p, T, &r_dual_t, &r_cent_t, &r_prim_t, &r_ieq_t);
            if (norm_t <= (1 - param.alpha * t) * norm) break;
            t *= param.beta;
        }
        x = x + @as(@Vector(n, T), @splat(t)) * x_step;
        if (init_mode == .infeasible) {
            slack = slack + @as(@Vector(m, T), @splat(t)) * slack_step;
        }
        ieq_dual = ieq_dual + @as(@Vector(m, T), @splat(t)) * ieq_dual_step;
        eq_dual = eq_dual + @as(@Vector(p, T), @splat(t)) * eq_dual_step;
    }
    return .{ .x = x, .ieq_dual = ieq_dual, .eq_dual = eq_dual };
}

test "optimizeGeneralConstraints QP" {
    const eq_constraints = EqConstraints(2, 1, f64){
        .a = zla.Mat(f64, 1, 2).init(.{ 1.0, 1.0 }),
        .b = .{1.0},
    };

    const ieq_constraints = InequalityConstraints(2, 2, f64){
        .f = struct {
            fn f(x: @Vector(2, f64)) @Vector(2, f64) {
                return .{ 0.8 - x[0], -x[1] };
            }
        }.f,
        .jac = struct {
            fn jac(x: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
                _ = x;
                return zla.Mat(f64, 2, 2).init(.{
                    -1.0, 0.0,
                    0.0,  -1.0,
                });
            }
        }.jac,
        .hess_sum = struct {
            fn hessSum(x: @Vector(2, f64), ieq_dual: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
                _ = x;
                _ = ieq_dual;
                return zla.Mat(f64, 2, 2).init(.{
                    0.0, 0.0,
                    0.0, 0.0,
                });
            }
        }.hessSum,
    };

    const solver = struct {
        fn solve(
            block_hess: *const zla.Mat(f64, 2, 2),
            block_h_jac_transpose: *const zla.Mat(f64, 2, 2),
            block_a: *const zla.Mat(f64, 1, 2),
            block_h_jac_diag_iedual: *const zla.Mat(f64, 2, 2),
            block_diag_h: *const @Vector(2, f64),
            r_dual: *const @Vector(2, f64),
            r_cent: *const @Vector(2, f64),
            r_prim: *const @Vector(1, f64),
            x_step: *@Vector(2, f64),
            ieq_dual_step: *@Vector(2, f64),
            eq_dual_step: *@Vector(1, f64),
        ) void {
            const block_a_transpose = block_a.transpose();
            const block_diag_h_mat = zla.Mat(f64, 2, 2).init(.{
                block_diag_h.*[0], 0.0,
                0.0,               block_diag_h.*[1],
            });
            var kkt = zla.Mat(f64, 5, 5).init(.{
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            });
            kkt.set_block(0, 0, block_hess);
            kkt.set_block(0, 2, block_h_jac_transpose);
            kkt.set_block(0, 4, &block_a_transpose);
            kkt.set_block(2, 0, block_h_jac_diag_iedual);
            kkt.set_block(2, 2, &block_diag_h_mat);
            kkt.set_block(4, 0, block_a);

            const rhs: @Vector(5, f64) = .{ -r_dual.*[0], -r_dual.*[1], -r_cent.*[0], -r_cent.*[1], -r_prim.*[0] };
            var step: @Vector(5, f64) = undefined;
            kkt.solve_lu(&rhs, &step) catch unreachable;

            x_step.* = .{ step[0], step[1] };
            ieq_dual_step.* = .{ step[2], step[3] };
            eq_dual_step.* = .{step[4]};
        }
    }.solve;

    const feasible_result = optimizeGeneralConstraints(2, 2, 1, f64, .feasible, f_qp, df, hessian, eq_constraints, ieq_constraints, solver, .{ 0.9, 0.2 }, .{ .e_feasible = 1e-9, .e_gap = 1e-9 });
    const infeasible_result = optimizeGeneralConstraints(2, 2, 1, f64, .infeasible, f_qp, df, hessian, eq_constraints, ieq_constraints, solver, .{ 0.0, 0.0 }, .{ .e_feasible = 1e-9, .e_gap = 1e-9 });
    inline for (.{ feasible_result, infeasible_result }) |result| {
        try std.testing.expectApproxEqAbs(0.8, result.x[0], 1e-6);
        try std.testing.expectApproxEqAbs(0.2, result.x[1], 1e-6);
        try std.testing.expectApproxEqAbs(1.0, result.x[0] + result.x[1], 1e-9);
        try std.testing.expect(result.x[0] >= 0.8 - 1e-9);
        try std.testing.expect(result.x[1] >= -1e-9);
        try std.testing.expectApproxEqAbs(1.4, result.ieq_dual[0], 1e-5);
        try std.testing.expectApproxEqAbs(0.0, result.ieq_dual[1], 1e-5);
        try std.testing.expectApproxEqAbs(-0.4, result.eq_dual[0], 1e-5);
    }
}
