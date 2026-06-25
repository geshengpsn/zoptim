const std = @import("std");
const zla = @import("zla");

pub fn EqConstraints(comptime n: usize, comptime p: usize, comptime T: type) type {
    return struct {
        a: zla.Mat(T, p, n),
        b: @Vector(p, T),
    };
}

pub fn Params(comptime T: type) type {
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

pub fn EqConstraintsResult(comptime n: usize, comptime p: usize, comptime T: type) type {
    return struct {
        x: @Vector(n, T),
        dual: @Vector(p, T),
    };
}

pub fn optimizeEqConstraints(
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

fn eqc_general_solver(
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

test "optimizeEqConstraints QP" {
    const eq_constraints = EqConstraints(2, 1, f64){
        .a = zla.Mat(f64, 1, 2).init(.{ 1.0, 1.0 }),
        .b = .{1.0},
    };
    const result = optimizeEqConstraints(2, 1, f64, f_qp, df, hessian, eq_constraints, eqc_general_solver, .{ 0.0, 0.0 }, .{ .e = 1e-10 });

    try std.testing.expectApproxEqAbs(1.0 / 3.0, result.x[0], 1e-9);
    try std.testing.expectApproxEqAbs(2.0 / 3.0, result.x[1], 1e-9);
    try std.testing.expectApproxEqAbs(1.0, result.x[0] + result.x[1], 1e-9);
    try std.testing.expectApproxEqAbs(-4.0 / 3.0, result.dual[0], 1e-9);
}
