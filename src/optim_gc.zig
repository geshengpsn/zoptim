const std = @import("std");
const zla = @import("zla");
const eqc = @import("optim_eqc.zig");
const EqConstraints = eqc.EqConstraints;

pub fn InequalityConstraints(comptime n: usize, comptime m: usize, comptime T: type) type {
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

pub fn GeneralConstraintsParams(comptime T: type) type {
    return struct {
        e_feasible: T = 1e-9,
        e_gap: T = 1e-9,
        miu: T = 10.0,
        alpha: T = 0.1,
        beta: T = 0.5,
    };
}

pub fn GeneralConstraintsResult(comptime n: usize, comptime m: usize, comptime p: usize, comptime T: type) type {
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

pub fn optimizeGeneralConstraints(
    comptime n: usize,
    comptime m: usize,
    comptime p: usize,
    comptime T: type,
    comptime init_mode: GeneralConstraintsInitMode,
    comptime use_correction_step: bool,
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

        if (comptime use_correction_step and init_mode == .infeasible) {
            const x_full_step = x + x_step;
            const slack_full_step = slack + slack_step;
            const h_full_step = ieq_constraints.f(x_full_step);
            const correction_r_ieq = h_full_step + slack_full_step;

            var correction_x_step: @Vector(n, T) = undefined;
            var correction_ieq_dual_step: @Vector(m, T) = undefined;
            var correction_eq_dual_step: @Vector(p, T) = undefined;
            const correction_r_dual: @Vector(n, T) = @splat(0);
            const correction_r_prim: @Vector(p, T) = @splat(0);
            const correction_r_cent = -ieq_dual * correction_r_ieq;
            solver(
                &block_hess,
                &block_h_jac_transpose,
                &eq_constraints.a,
                &block_h_jac_diag_ieq_dual,
                &slack,
                &correction_r_dual,
                &correction_r_cent,
                &correction_r_prim,
                &correction_x_step,
                &correction_ieq_dual_step,
                &correction_eq_dual_step,
            );

            var correction_slack_step: @Vector(m, T) = undefined;
            h_jac_val.vec_mul(&correction_x_step, &correction_slack_step);
            correction_slack_step = -correction_r_ieq - correction_slack_step;

            x_step += correction_x_step;
            slack_step += correction_slack_step;
            ieq_dual_step += correction_ieq_dual_step;
            eq_dual_step += correction_eq_dual_step;
        }

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

fn gc_general_solver(
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

    const feasible_result = optimizeGeneralConstraints(
        2,
        2,
        1,
        f64,
        .feasible,
        false,
        f_qp,
        df,
        hessian,
        eq_constraints,
        ieq_constraints,
        gc_general_solver,
        .{ 0.9, 0.2 },
        .{ .e_feasible = 1e-9, .e_gap = 1e-9 },
    );
    const infeasible_result = optimizeGeneralConstraints(
        2,
        2,
        1,
        f64,
        .infeasible,
        true,
        f_qp,
        df,
        hessian,
        eq_constraints,
        ieq_constraints,
        gc_general_solver,
        .{ 0.0, 0.0 },
        .{ .e_feasible = 1e-9, .e_gap = 1e-9 },
    );
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
