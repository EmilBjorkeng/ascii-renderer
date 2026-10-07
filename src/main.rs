mod model;

use model::{FACE_INDICES, FACE_NORMALS, VERTEX_COUNT, VERTICES};

use glam::{Mat3, Vec2, Vec3};

use std::f32::consts::TAU;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

const SCREEN_W: usize = 80;
const SCREEN_H: usize = 40;
const CHAR_ASPECT: f32 = 2.0;

const FOV: f32 = 60.0;
const CAMERA_DISTANCE: f32 = 2.8;

const SPEED_X: f32 = 0.04;
const SPEED_Y: f32 = 0.03;

const FPS: u64 = 30;
const FRAME_TIME: Duration = Duration::from_nanos(1_000_000_000 / FPS);

const AMBIENT: f32 = 0.15;
const SHADES: &[u8] = b".,-~:;=!*#$@";

type Screen = [[u8; SCREEN_W]; SCREEN_H];
type Depth = [[f32; SCREEN_W]; SCREEN_H];

/// 2D cross product
fn edge(a: Vec3, b: Vec3, px: f32, py: f32) -> f32 {
    (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x)
}

fn fill_triangle(screen: &mut Screen, depth: &mut Depth, a: Vec3, b: Vec3, c: Vec3, shade: u8) {
    // Skip triangles with (nearly) zero area
    let area = edge(a, b, c.x, c.y);
    if area.abs() < 1e-6 {
        return;
    }

    // Bounding box, clipped to the screen
    let min_x = a.x.min(b.x).min(c.x).floor().max(0.0);
    let max_x = a.x.max(b.x).max(c.x).ceil().min((SCREEN_W - 1) as f32);
    let min_y = a.y.min(b.y).min(c.y).floor().max(0.0);
    let max_y = a.y.max(b.y).max(c.y).ceil().min((SCREEN_H - 1) as f32);

    // Skip entirely off-screen triangles
    if min_x > max_x || min_y > max_y {
        return;
    }

    // Loop over bounding box
    for y in min_y as usize..=max_y as usize {
        for x in min_x as usize..=max_x as usize {
            // Pixel centre
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;

            // Barycentric weights
            let w0 = edge(b, c, px, py) / area;
            let w1 = edge(c, a, px, py) / area;
            let w2 = edge(a, b, px, py) / area;
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }

            // Draw if closer than anything drawn before (d is 1/z: larger = closer)
            let d = w0 * a.z + w1 * b.z + w2 * c.z;
            if d > depth[y][x] {
                depth[y][x] = d;
                screen[y][x] = shade;
            }
        }
    }
}

fn main() {
    let mut screen = [[b' '; SCREEN_W]; SCREEN_H]; // Screen buffer
    let mut depth = [[0.0_f32; SCREEN_W]; SCREEN_H]; // Depth buffer

    let focal_length = 1.0 / (FOV.to_radians() / 2.0).tan();
    let chars_per_world_unit = Vec2::new(CHAR_ASPECT, 1.0) * focal_length * SCREEN_H as f32 * 0.5;
    let screen_centre = Vec2::new(SCREEN_W as f32, SCREEN_H as f32) * 0.5;

    // The scene light
    let light = Vec3::new(-1.0, 1.0, 1.0).normalize();

    // Rotation angle of the object (in X and Y)
    let mut angle = Vec2::ZERO;

    // Vertices
    let mut world = [Vec3::ZERO; VERTEX_COUNT];
    let mut projected = [Vec3::ZERO; VERTEX_COUNT];

    // Lock stdout once (instead of on every write)
    // Create the buffer used to print the entire picture at once (frame)
    // +3 for the escape code, +1 per row for the newline
    let mut out = io::stdout().lock();
    let mut frame = Vec::with_capacity(3 + (SCREEN_W + 1) * SCREEN_H);

    // Clear the terminal
    print!("\x1b[2J");

    loop {
        // Clear the screen and depth buffer
        for row in &mut screen {
            row.fill(b' ');
        }
        for row in &mut depth {
            row.fill(0.0);
        }

        // Model matrix: RotY * RotX
        let model_matrix = Mat3::from_rotation_y(angle.y) * Mat3::from_rotation_x(angle.x);

        // Local to Screen space projection
        for (i, &v) in VERTICES.iter().enumerate() {
            let world_pos = model_matrix * Vec3::from(v);
            let view_z = CAMERA_DISTANCE - world_pos.z;
            world[i] = world_pos; // World space
            projected[i] = Vec3::new(
                screen_centre.x + world_pos.x * chars_per_world_unit.x / view_z,
                screen_centre.y - world_pos.y * chars_per_world_unit.y / view_z,
                1.0 / view_z,
            );
        }

        for (face, &normal) in FACE_INDICES.iter().zip(FACE_NORMALS.iter()) {
            let world_normal = model_matrix * Vec3::from(normal);

            // Backface culling
            let point = world[face[0] as usize];
            let to_camera = Vec3::new(0.0, 0.0, CAMERA_DISTANCE) - point;
            if world_normal.dot(to_camera) <= 0.0 {
                continue;
            }

            // Flat shading (one shade per face)
            let diffuse = world_normal.dot(light).max(0.0);
            let brightness = AMBIENT + (1.0 - AMBIENT) * diffuse;
            let level = ((brightness * SHADES.len() as f32) as usize).min(SHADES.len() - 1);
            let shade = SHADES[level]; // Get the shade (colour) of the current face

            // Split the face quad into triangles (0, 1, 2) and (0, 2, 3)
            // We know all faces have 4 vertices
            for k in 1..face.len() - 1 {
                fill_triangle(
                    &mut screen,
                    &mut depth,
                    projected[face[0] as usize],
                    projected[face[k] as usize],
                    projected[face[k + 1] as usize],
                    shade,
                );
            }
        }

        frame.clear();
        frame.extend_from_slice(b"\x1b[H"); // Cursor to top-left
        for row in &screen {
            frame.extend_from_slice(row);
            frame.push(b'\n');
        }

        // Send the whole frame to the terminal in one write
        out.write_all(&frame).unwrap();
        out.flush().unwrap();

        // Increase the angle
        angle = (angle + Vec2::new(SPEED_X, SPEED_Y)) % TAU;

        // Wait for the next frame
        thread::sleep(FRAME_TIME);
    }
}
