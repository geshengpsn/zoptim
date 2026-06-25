const std = @import("std");
const zla = @import("zla");

fn armijo(
    comptime n: usize,
    comptime T: type,
    func: fn (@Vector(n, T)) T,
    x: @Vector(n, T),
    grad: @Vector(n, T),
    dir: @Vector(n, T),
) T {
    var t: T = 1.0;
    const f_x = func(x);

    while (func(x + @as(@Vector(n, T), @splat(t)) * dir) > f_x + 0.4 * t * @reduce(.Add, grad * dir)) {
        t *= 0.5;
    }
    return t;
}

pub fn optimizeNoConstraints(
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

// for testing only
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

pub fn noc_general_solver(
    a: *const zla.Mat(f64, 2, 2),
    b: *const @Vector(2, f64),
    x: *@Vector(2, f64),
) void {
    const rhs = -b.*;
    a.solve_lu(&rhs, x) catch unreachable;
}

test "optimizeNoConstraints" {
    const init_x: @Vector(2, f64) = .{ 2.0, 4.0 };
    const result = optimizeNoConstraints(
        2,
        f64,
        funcs.f,
        funcs.grad,
        funcs.hess,
        noc_general_solver,
        init_x,
        1e-10,
    );
    try std.testing.expectApproxEqAbs(@reduce(.Add, funcs.grad(result)), 0.0, 1e-4);
}
