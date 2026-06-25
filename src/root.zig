//! By convention, root.zig is the root source file when making a package.
//!
const std = @import("std");
const Io = std.Io;
const zla = @import("zla");
const zplotly = @import("zplotly");
const noc = @import("optim_noc.zig");
const eqc = @import("optim_eqc.zig");
const gc = @import("optim_gc.zig");

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
}
