//! By convention, root.zig is the root source file when making a package.
//!
const std = @import("std");
const Io = std.Io;
const zla = @import("zla");
const zplotly = @import("zplotly");
const noc = @import("optim_noc.zig");
const eqc = @import("optim_eqc.zig");
const gc = @import("optim_gc.zig");
const root_find = @import("root_find.zig");

pub const findRoot = root_find.findRoot;
pub const RootParams = root_find.RootParams;
pub const RootResult = root_find.RootResult;
pub const RootError = root_find.RootError;

pub const optimizeNoConstraints = noc.optimizeNoConstraints;
pub const noc_general_solver = noc.noc_general_solver;

pub const EqConstraints = eqc.EqConstraints;
pub const EqParams = eqc.Params;
pub const optimizeEqConstraints = eqc.optimizeEqConstraints;
pub const EqConstraintsResult = eqc.EqConstraintsResult;
pub const eqc_general_solver = eqc.eqc_general_solver;

pub const InequalityConstraints = gc.InequalityConstraints;
pub const GeneralConstraintsParams = gc.GeneralConstraintsParams;
pub const GeneralConstraintsResult = gc.GeneralConstraintsResult;
pub const optimizeGeneralConstraints = gc.optimizeGeneralConstraints;
pub const gc_general_solver = gc.gc_general_solver;

test {
    _ = noc;
    _ = eqc;
    _ = gc;
    _ = root_find;
}

test "public root finding interface" {
    const funcs = struct {
        fn f(x: @Vector(1, f64)) @Vector(1, f64) {
            return .{x[0] * x[0] - 2};
        }

        fn jac(x: @Vector(1, f64)) zla.Mat(f64, 1, 1) {
            return zla.Mat(f64, 1, 1).init(.{2 * x[0]});
        }
    };
    const params: RootParams(f64) = .{ .e = 1e-12 };
    const pending: RootError!RootResult(1, f64) = findRoot(1, f64, funcs.f, funcs.jac, .{1}, params);
    const result = try pending;
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2)), result.x[0], params.e);
}
