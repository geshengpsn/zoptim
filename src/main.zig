const std = @import("std");
const Io = std.Io;

const zoptim = @import("zoptim");

pub fn main(init: std.process.Init) !void {
    const arena: std.mem.Allocator = init.arena.allocator();
    const io = init.io;

    try zoptim.plotNewtonArmijo(arena, io);
}
