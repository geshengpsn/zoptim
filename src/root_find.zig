const std = @import("std");
const zla = @import("zla");

pub const RootError = error{
    InvalidParameters,
    NonFiniteValue,
    SingularJacobian,
    LineSearchFailed,
    MaxIterationsExceeded,
};

pub fn RootParams(comptime T: type) type {
    return struct {
        /// Absolute tolerance for max_i |f(x)[i]|.
        e: T = @max(1e-9, 16 * std.math.floatEps(T)),
        /// Required residual decrease, in (0, 0.5).
        alpha: T = 0.4,
        /// Step reduction factor, in (0, 1).
        beta: T = 0.5,
        /// Maximum Newton steps; zero only accepts an initially converged point.
        max_iterations: usize = 100,
        /// Maximum step reductions per iteration; zero tries only the full step.
        max_backtracks: usize = 50,
    };
}

pub fn RootResult(comptime n: usize, comptime T: type) type {
    return struct {
        x: @Vector(n, T),
        /// Infinity norm of f(x), at most the requested tolerance.
        residual_norm: T,
        /// Number of accepted Newton steps.
        iterations: usize,
    };
}

fn infinityNorm(comptime n: usize, comptime T: type, value: @Vector(n, T)) T {
    var norm: T = 0;
    inline for (0..n) |i| {
        if (!std.math.isFinite(value[i])) return std.math.inf(T);
        norm = @max(norm, @abs(value[i]));
    }
    return norm;
}

/// The single struct parameter type of `f`.
fn paramStruct(comptime F: type) type {
    return @typeInfo(F).@"fn".params[0].type.?;
}

/// The return type of `f`.
fn returnVector(comptime F: type) type {
    return @typeInfo(F).@"fn".return_type.?;
}

fn fieldVectorLen(comptime Vec: type) usize {
    return @typeInfo(Vec).vector.len;
}

/// The vector type solved by `findRootPartial` for the given struct.
fn freeVector(comptime Args: type, comptime free_index: usize) type {
    return std.meta.fields(Args)[free_index].type;
}

/// The floating point element type shared by every struct field.
fn scalarType(comptime Args: type) type {
    return @typeInfo(std.meta.fields(Args)[0].type).vector.child;
}

/// The square Jacobian type df/d(free vector).
fn freeJacobian(comptime Args: type, comptime free_index: usize) type {
    const m = fieldVectorLen(freeVector(Args, free_index));
    return zla.Mat(scalarType(Args), m, m);
}

fn validateStruct(comptime Args: type, comptime free_index: usize) void {
    comptime {
        const fields = std.meta.fields(Args);
        if (fields.len == 0) @compileError("findRootPartial requires a non-empty struct");
        if (free_index >= fields.len) @compileError("findRootPartial free_index out of range");
        if (@typeInfo(fields[0].type) != .vector) @compileError("findRootPartial struct fields must be @Vector");
        const T = @typeInfo(fields[0].type).vector.child;
        if (@typeInfo(T) != .float) @compileError("findRootPartial requires floating point struct fields");
        for (fields) |field| {
            if (@typeInfo(field.type) != .vector) @compileError("findRootPartial struct fields must be @Vector");
            if (@typeInfo(field.type).vector.child != T) @compileError("findRootPartial struct fields must share one element type");
        }
    }
}

/// Solves f(x) = 0 using damped Newton steps and residual backtracking.
/// `jac` must return J[i,j] = df[i]/dx[j]; each step solves J * step = -f(x)
/// with row scaling and zla's pivoted LU solver. Pass null for default parameters.
/// Convergence is local and depends on the initial guess and Jacobian.
/// Non-finite initial values, residuals, Jacobians, or steps return
/// NonFiniteValue; non-finite trial points/residuals trigger backtracking.
pub fn findRoot(
    comptime n: usize,
    comptime T: type,
    f: fn (x: @Vector(n, T)) @Vector(n, T),
    jac: fn (x: @Vector(n, T)) zla.Mat(T, n, n),
    init_x: @Vector(n, T),
    params: ?RootParams(T),
) RootError!RootResult(n, T) {
    comptime {
        if (n == 0) @compileError("findRoot requires at least one equation");
        if (@typeInfo(T) != .float) @compileError("findRoot requires a floating point type");
    }
    const param: RootParams(T) = params orelse .{};
    if (!std.math.isFinite(param.e) or param.e <= 0 or
        !(param.alpha > 0 and param.alpha < 0.5) or
        !(param.beta > 0 and param.beta < 1))
    {
        return error.InvalidParameters;
    }

    const Vec = @Vector(n, T);
    var x = init_x;
    if (!std.math.isFinite(infinityNorm(n, T, x))) return error.NonFiniteValue;
    var residual = f(x);
    var norm = infinityNorm(n, T, residual);
    if (!std.math.isFinite(norm)) return error.NonFiniteValue;

    var iterations: usize = 0;
    while (norm > param.e) : (iterations += 1) {
        if (iterations == param.max_iterations) return error.MaxIterationsExceeded;
        var jacobian = jac(x);
        var rhs = -residual;
        // Make LU's absolute pivot threshold relative to each equation's scale.
        inline for (0..n) |i| {
            const row = jacobian.getRow(i);
            const scale = infinityNorm(n, T, row);
            if (!std.math.isFinite(scale)) return error.NonFiniteValue;
            if (scale == 0) return error.SingularJacobian;
            jacobian.setRow(i, row / @as(Vec, @splat(scale)));
            rhs[i] /= scale;
        }
        if (!std.math.isFinite(infinityNorm(n, T, rhs))) return error.NonFiniteValue;
        var step: Vec = undefined;
        jacobian.solveLuAssign(&rhs, &step) catch return error.SingularJacobian;
        if (!std.math.isFinite(infinityNorm(n, T, step))) return error.NonFiniteValue;

        var t: T = 1;
        var backtracks: usize = 0;
        while (true) : (backtracks += 1) {
            const trial_x = x + @as(Vec, @splat(t)) * step;
            if (@reduce(.And, trial_x == x)) return error.LineSearchFailed;
            if (std.math.isFinite(infinityNorm(n, T, trial_x))) {
                const trial_residual = f(trial_x);
                const trial_norm = infinityNorm(n, T, trial_residual);
                if (trial_norm <= param.e or
                    (trial_norm < norm and trial_norm <= (1 - param.alpha * t) * norm))
                {
                    x = trial_x;
                    residual = trial_residual;
                    norm = trial_norm;
                    break;
                }
            }
            if (backtracks == param.max_backtracks) return error.LineSearchFailed;
            t *= param.beta;
        }
    }
    return .{ .x = x, .residual_norm = norm, .iterations = iterations };
}

fn StructRootResult(comptime Args: type, comptime T: type) type {
    return struct {
        args: Args,
        residual_norm: T,
        iterations: usize,
    };
}

fn structIsFinite(comptime Args: type, comptime T: type, args: Args) bool {
    inline for (std.meta.fields(Args)) |field| {
        if (!std.math.isFinite(infinityNorm(fieldVectorLen(field.type), T, @field(args, field.name)))) return false;
    }
    return true;
}

fn solveStructPartial(
    comptime Args: type,
    comptime free_index: usize,
    comptime T: type,
    f: fn (Args) freeVector(Args, free_index),
    jac: fn (Args) freeJacobian(Args, free_index),
    init: Args,
    param: RootParams(T),
) RootError!StructRootResult(Args, T) {
    const Vec = freeVector(Args, free_index);
    const m = comptime fieldVectorLen(Vec);
    const free_name = std.meta.fields(Args)[free_index].name;

    var args = init;
    if (!structIsFinite(Args, T, args)) return error.NonFiniteValue;
    var residual = f(args);
    var norm = infinityNorm(m, T, residual);
    if (!std.math.isFinite(norm)) return error.NonFiniteValue;

    var iterations: usize = 0;
    while (norm > param.e) : (iterations += 1) {
        if (iterations == param.max_iterations) return error.MaxIterationsExceeded;
        var jacobian = jac(args);
        var rhs = -residual;
        // Make LU's absolute pivot threshold relative to each equation's scale.
        inline for (0..m) |i| {
            const row = jacobian.getRow(i);
            const scale = infinityNorm(m, T, row);
            if (!std.math.isFinite(scale)) return error.NonFiniteValue;
            if (scale == 0) return error.SingularJacobian;
            jacobian.setRow(i, row / @as(Vec, @splat(scale)));
            rhs[i] /= scale;
        }
        if (!std.math.isFinite(infinityNorm(m, T, rhs))) return error.NonFiniteValue;
        var step: Vec = undefined;
        jacobian.solveLuAssign(&rhs, &step) catch return error.SingularJacobian;
        if (!std.math.isFinite(infinityNorm(m, T, step))) return error.NonFiniteValue;

        var t: T = 1;
        var backtracks: usize = 0;
        while (true) : (backtracks += 1) {
            const free = @field(args, free_name) + @as(Vec, @splat(t)) * step;
            var trial = args;
            @field(trial, free_name) = free;
            if (@reduce(.And, free == @field(args, free_name))) return error.LineSearchFailed;
            if (structIsFinite(Args, T, trial)) {
                const trial_residual = f(trial);
                const trial_norm = infinityNorm(m, T, trial_residual);
                if (trial_norm <= param.e or
                    (trial_norm < norm and trial_norm <= (1 - param.alpha * t) * norm))
                {
                    args = trial;
                    residual = trial_residual;
                    norm = trial_norm;
                    break;
                }
            }
            if (backtracks == param.max_backtracks) return error.LineSearchFailed;
            t *= param.beta;
        }
    }
    return .{ .args = args, .residual_norm = norm, .iterations = iterations };
}

/// Solves f(args) = 0 for one field of a struct of vectors while the remaining
/// fields stay fixed. `f` and `jac` each take the struct; `f` returns the free
/// `@Vector` and `jac` returns its square Jacobian. `free_field` selects which
/// field is solved; the other fields of `init` are constants.
///
/// All concrete types (struct shape, element type, dimensions) are derived at
/// compile time from `f`, so callers only pass the functions, the initial
/// struct and the free field. Pass null for default parameters.
pub fn findRootPartial(
    f: anytype,
    comptime free_field: std.meta.FieldEnum(paramStruct(@TypeOf(f))),
    jac: fn (paramStruct(@TypeOf(f))) freeJacobian(paramStruct(@TypeOf(f)), @intFromEnum(free_field)),
    init: paramStruct(@TypeOf(f)),
    params: ?RootParams(scalarType(paramStruct(@TypeOf(f)))),
) RootError!paramStruct(@TypeOf(f)) {
    const Args = paramStruct(@TypeOf(f));
    const T = scalarType(Args);
    const free_index = comptime @intFromEnum(free_field);
    comptime {
        if (@typeInfo(@TypeOf(f)).@"fn".params.len != 1) {
            @compileError("findRootPartial: f must take exactly one struct parameter");
        }
        validateStruct(Args, free_index);
        if (returnVector(@TypeOf(f)) != freeVector(Args, free_index)) {
            @compileError("findRootPartial: f must return the free vector type");
        }
    }
    const param: RootParams(T) = params orelse .{};
    if (!std.math.isFinite(param.e) or param.e <= 0 or
        !(param.alpha > 0 and param.alpha < 0.5) or
        !(param.beta > 0 and param.beta < 1))
    {
        return error.InvalidParameters;
    }
    const result = try solveStructPartial(Args, free_index, T, f, jac, init, param);
    return result.args;
}

test "findRoot solves a nonlinear system with f32 and f64" {
    inline for (.{ f32, f64 }) |T| {
        const funcs = struct {
            fn f(x: @Vector(2, T)) @Vector(2, T) {
                return .{ x[0] * x[0] + x[1] - 5, 2 * x[0] - x[1] - 3 };
            }

            fn jac(x: @Vector(2, T)) zla.Mat(T, 2, 2) {
                return zla.Mat(T, 2, 2).init(.{
                    2 * x[0], 1,
                    2,        -1,
                });
            }
        };
        const result = try findRoot(2, T, funcs.f, funcs.jac, .{ 1, 0 }, null);
        const params: RootParams(T) = .{};
        try std.testing.expectApproxEqAbs(@as(T, 2), result.x[0], params.e);
        try std.testing.expectApproxEqAbs(@as(T, 1), result.x[1], params.e);
        try std.testing.expect(result.residual_norm <= params.e);
        try std.testing.expectEqual(infinityNorm(2, T, funcs.f(result.x)), result.residual_norm);
        try std.testing.expect(result.iterations > 0);
    }
}

test "findRoot accepts an initial root without evaluating the Jacobian" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return .{x[0] - 2};
        }

        fn jac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            unreachable;
        }
    };
    const result = try findRoot(1, f64, funcs.f, funcs.jac, .{2}, .{ .max_iterations = 0 });
    try std.testing.expectEqual(@as(f64, 2), result.x[0]);
    try std.testing.expectEqual(@as(f64, 0), result.residual_norm);
    try std.testing.expectEqual(@as(usize, 0), result.iterations);
}

test "findRoot scales equations and pivots the Jacobian" {
    inline for (.{ f32, f64 }) |T| {
        const funcs = struct {
            const scale: T = if (T == f32) 0x1p-40 else 0x1p-400;

            fn f(x: @Vector(2, T)) @Vector(2, T) {
                return .{ scale * (x[1] - 20), (x[0] - 10) / scale };
            }

            fn jac(_: @Vector(2, T)) zla.Mat(T, 2, 2) {
                return zla.Mat(T, 2, 2).init(.{ 0, scale, 1 / scale, 0 });
            }
        };
        const result = try findRoot(2, T, funcs.f, funcs.jac, .{ 0, 0 }, null);
        try std.testing.expectEqual(@as(@Vector(2, T), .{ 10, 20 }), result.x);
        try std.testing.expectEqual(@as(T, 0), result.residual_norm);
        try std.testing.expectEqual(@as(usize, 1), result.iterations);
    }
}

test "findRoot backtracks through non-finite trial residuals" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return .{@log(x[0])};
        }

        fn jac(x: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1 / x[0]});
        }
    };
    try std.testing.expectError(error.LineSearchFailed, findRoot(1, f64, funcs.f, funcs.jac, .{10}, .{ .max_backtracks = 0 }));
    const result = try findRoot(1, f64, funcs.f, funcs.jac, .{10}, .{ .e = 1e-12 });
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.x[0], 1e-12);
    try std.testing.expect(result.residual_norm <= 1e-12);
}

test "findRoot enforces the iteration limit and accepts its final step" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return .{x[0] - 2};
        }

        fn jac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1});
        }

        fn slowJac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{2});
        }
    };
    try std.testing.expectError(error.MaxIterationsExceeded, findRoot(1, f64, funcs.f, funcs.jac, .{0}, .{ .max_iterations = 0 }));
    try std.testing.expectError(error.MaxIterationsExceeded, findRoot(1, f64, funcs.f, funcs.slowJac, .{0}, .{ .max_iterations = 1 }));
    const result = try findRoot(1, f64, funcs.f, funcs.jac, .{0}, .{ .max_iterations = 1, .max_backtracks = 0 });
    try std.testing.expectEqual(@as(f64, 2), result.x[0]);
    try std.testing.expectEqual(@as(usize, 1), result.iterations);
}

test "findRoot reports a singular Jacobian" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return .{x[0] * x[0] - 2};
        }

        fn jac(x: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{2 * x[0]});
        }
    };
    try std.testing.expectError(error.SingularJacobian, findRoot(1, f64, funcs.f, funcs.jac, .{0}, null));
}

test "findRoot rejects invalid parameters before evaluating callbacks" {
    const funcs = struct {
        fn f(_: @Vector(1, f64)) @Vector(1, f64) {
            unreachable;
        }

        fn jac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            unreachable;
        }
    };
    const invalid = [_]RootParams(f64){
        .{ .e = 0 },
        .{ .e = -1 },
        .{ .e = std.math.inf(f64) },
        .{ .e = std.math.nan(f64) },
        .{ .alpha = 0 },
        .{ .alpha = 0.5 },
        .{ .alpha = std.math.nan(f64) },
        .{ .beta = 0 },
        .{ .beta = 1 },
        .{ .beta = std.math.inf(f64) },
        .{ .beta = std.math.nan(f64) },
    };
    for (invalid) |params| {
        try std.testing.expectError(error.InvalidParameters, findRoot(1, f64, funcs.f, funcs.jac, .{0}, params));
    }
}

test "findRoot rejects non-finite inputs residuals Jacobians and steps" {
    inline for (.{ std.math.inf(f64), std.math.nan(f64) }) |invalid| {
        const funcs = struct {
            fn f(x: @Vector(2, f64)) @Vector(2, f64) {
                return x;
            }

            fn badF(_: @Vector(2, f64)) @Vector(2, f64) {
                return .{ invalid, 1 };
            }

            fn jac(_: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
                return zla.Mat(f64, 2, 2).init(.{ 1e-5, 0, 0, 1 });
            }

            fn badJac(_: @Vector(2, f64)) zla.Mat(f64, 2, 2) {
                return zla.Mat(f64, 2, 2).init(.{ 1, invalid, 0, 1 });
            }
        };
        try std.testing.expectError(error.NonFiniteValue, findRoot(2, f64, funcs.f, funcs.jac, .{ invalid, 1 }, null));
        try std.testing.expectError(error.NonFiniteValue, findRoot(2, f64, funcs.badF, funcs.jac, .{ 1, 1 }, null));
        try std.testing.expectError(error.NonFiniteValue, findRoot(2, f64, funcs.f, funcs.badJac, .{ 1, 1 }, null));
        try std.testing.expectError(error.NonFiniteValue, findRoot(2, f64, funcs.f, funcs.jac, .{ 1e308, 1 }, null));
    }
}

test "findRoot handles large and small residuals without squaring" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return x;
        }

        fn jac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1});
        }
    };
    for ([_]f64{ 1e200, 1e-200 }) |initial| {
        const result = try findRoot(1, f64, funcs.f, funcs.jac, .{initial}, .{ .e = 1e-210 });
        try std.testing.expectEqual(@as(f64, 0), result.x[0]);
        try std.testing.expectEqual(@as(usize, 1), result.iterations);
    }
}

test "findRoot reports failed backtracking and stagnation" {
    const funcs = struct {
        fn f(_: @Vector(1, f64)) @Vector(1, f64) {
            return .{1};
        }

        fn jac(_: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1});
        }
    };
    try std.testing.expectError(error.LineSearchFailed, findRoot(1, f64, funcs.f, funcs.jac, .{0}, .{ .max_backtracks = 3 }));
    try std.testing.expectError(error.LineSearchFailed, findRoot(1, f64, funcs.f, funcs.jac, .{1e30}, null));
}

test "findRootPartial holds x1 fixed and solves x2" {
    const Args = struct {
        x1: @Vector(1, f64),
        x2: @Vector(1, f64),
    };
    const funcs = struct {
        fn f(args: Args) @Vector(1, f64) {
            return .{args.x1[0] + args.x2[0] - 3};
        }

        fn jac(_: Args) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1});
        }
    };
    const result = try findRootPartial(funcs.f, .x2, funcs.jac, .{ .x1 = .{1}, .x2 = .{0} }, null);
    try std.testing.expectEqual(@as(f64, 1), result.x1[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.x2[0], 1e-12);
}

test "findRootPartial solves a nonlinear free field with f32 and f64" {
    inline for (.{ f32, f64 }) |T| {
        const Args = struct {
            x1: @Vector(1, T),
            x2: @Vector(1, T),
        };
        const funcs = struct {
            fn f(args: Args) @Vector(1, T) {
                return .{args.x2[0] * args.x2[0] - args.x1[0]};
            }

            fn jac(args: Args) zla.Mat(T, 1, 1) {
                return zla.Mat(T, 1, 1).init(.{2 * args.x2[0]});
            }
        };
        const result = try findRootPartial(funcs.f, .x2, funcs.jac, .{ .x1 = .{4}, .x2 = .{1} }, null);
        try std.testing.expectEqual(@as(T, 4), result.x1[0]);
        try std.testing.expectApproxEqAbs(@as(T, 2), result.x2[0], @sqrt(std.math.floatEps(T)));
    }
}

test "findRootPartial solves a free vector and leaves other struct fields fixed" {
    const Args = struct {
        x: @Vector(2, f64),
        p: @Vector(1, f64),
    };
    const funcs = struct {
        fn f(args: Args) @Vector(2, f64) {
            return .{ args.x[0] + args.x[1] - args.p[0], args.x[0] - args.x[1] };
        }

        fn jac(_: Args) zla.Mat(f64, 2, 2) {
            return zla.Mat(f64, 2, 2).init(.{ 1, 1, 1, -1 });
        }
    };
    const result = try findRootPartial(funcs.f, .x, funcs.jac, .{ .x = .{ 0, 0 }, .p = .{4} }, null);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.x[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.x[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 4), result.p[0]);
}

test "findRootPartial rejects a non-finite fixed field" {
    const Args = struct {
        x1: @Vector(1, f64),
        x2: @Vector(1, f64),
    };
    const funcs = struct {
        fn f(args: Args) @Vector(1, f64) {
            return .{args.x1[0] + args.x2[0] - 3};
        }

        fn jac(_: Args) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{1});
        }
    };
    try std.testing.expectError(error.NonFiniteValue, findRootPartial(funcs.f, .x2, funcs.jac, .{ .x1 = .{std.math.inf(f64)}, .x2 = .{0} }, null));
}

test "findRootPartial rejects invalid parameters before evaluating callbacks" {
    const Args = struct {
        x1: @Vector(1, f64),
        x2: @Vector(1, f64),
    };
    const funcs = struct {
        fn f(_: Args) @Vector(1, f64) {
            unreachable;
        }

        fn jac(_: Args) zla.Mat(f64, 1, 1) {
            unreachable;
        }
    };
    try std.testing.expectError(error.InvalidParameters, findRootPartial(funcs.f, .x2, funcs.jac, .{ .x1 = .{1}, .x2 = .{0} }, .{ .e = 0 }));
}
