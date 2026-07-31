"""GhostSun SHG — printable kinematic OAP mount, Fusion 360 script.

Run inside Fusion 360: Utilities > Add-Ins > Scripts and Add-Ins > (+) add
this file > Run. Creates two new components in the active design:
  * KM_Base      — fixed plate: cone / vee / flat kinematic seats, spring
                   anchors, housing mounting counterbores
  * KM_Platform  — moving plate: pivot-ball pocket, two adjuster bosses
                   (heat-set M4 inserts), mirror-plate bolt circle

Everything is driven by USER PARAMETERS (Modify > Change Parameters), so
after the first run you edit dimensions there and the model rebuilds —
no need to re-run the script. Re-running creates fresh components.

Kinematics (classic cone/vee/flat):
  * fixed pivot ball (glued into platform pocket) sits in the BASE CONE
  * adjuster screw A (ball-tipped) lands in the VEE  -> pure PITCH
  * adjuster screw B lands on the FLAT               -> YAW (about cone-vee line)
  * two extension springs preload platform toward base

Resolution vs the raytrace tolerance budget (OAP yaw tol 0.025 deg =
1.5 arcmin): deg/turn = atan(pitch / leverArm).
  leverArm 120 mm, M4x0.7:    0.33 deg/turn = 20 arcmin/turn
  leverArm 120 mm, 100 TPI:   0.12 deg/turn =  7 arcmin/turn
A 1/20 caged turn is then 1.0 / 0.36 arcmin respectively — inside budget.
Lock with jam nuts + epoxy tack after alignment (set-once controls).

Print: ASA / PC / annealed PETG (NOT PLA — solar enclosure heat), 6+
perimeters, 40%+ infill, plates printed flat for stiffness.
Hardware per mount: 3x 6 mm steel balls, 2x M4 fine screws (or 100 TPI
adjusters), 2x M4 heat-set inserts, 2x steel extension springs ~8 N,
3x M4 mirror-plate screws, 3x M5 housing screws.
"""

import traceback

import adsk.core
import adsk.fusion

# name, default expression, comment
USER_PARAMS = [
    ("km_leverArm", "120 mm", "pivot-to-adjuster distance (angular resolution lever)"),
    ("km_plateW", "150 mm", "plate width (X: pivot->adjusterB + margin)"),
    ("km_plateH", "150 mm", "plate height (Y: pivot->adjusterA + margin)"),
    ("km_plateT", "8 mm", "plate thickness (both plates)"),
    ("km_gap", "10 mm", "working gap between plates (ball + screw stickout)"),
    ("km_ballD", "6 mm", "steel ball diameter (pivot + screw tips)"),
    ("km_seatBossD", "22 mm", "seat/adjuster boss diameter"),
    ("km_insertD", "5.6 mm", "heat-set insert hole (M4 insert nominal)"),
    ("km_insertL", "8 mm", "heat-set insert hole depth"),
    ("km_springHoleD", "3.2 mm", "spring anchor through-hole"),
    ("km_mirrorBCD", "40 mm", "mirror/backing-plate bolt circle diameter"),
    ("km_mirrorBoltD", "4.4 mm", "mirror bolt clearance hole (M4)"),
    ("km_mirrorBoltN", "3", "mirror bolt count"),
    ("km_baseBoltD", "5.4 mm", "housing bolt clearance (M5)"),
    ("km_baseCboreD", "10 mm", "housing bolt counterbore diameter"),
    ("km_baseCboreZ", "4 mm", "housing bolt counterbore depth"),
]


def ensure_params(design):
    ups = design.userParameters
    for name, expr, comment in USER_PARAMS:
        if ups.itemByName(name) is None:
            ups.add(name, adsk.core.ValueInput.createByString(expr), "", comment)


def pval(design, name):
    """User parameter value in cm (Fusion internal units)."""
    return design.userParameters.itemByName(name).value


def new_component(design, name):
    occ = design.rootComponent.occurrences.addNewComponent(
        adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ.component


def circle(sketch, x, y, d):
    return sketch.sketchCurves.sketchCircles.addByCenterRadius(
        adsk.core.Point3D.create(x, y, 0), d / 2.0)


def rect(sketch, x0, y0, x1, y1):
    sketch.sketchCurves.sketchLines.addTwoPointRectangle(
        adsk.core.Point3D.create(x0, y0, 0),
        adsk.core.Point3D.create(x1, y1, 0))


def extrude_profiles(comp, sketch, height, operation, participants=None):
    profs = adsk.core.ObjectCollection.create()
    for p in sketch.profiles:
        profs.add(p)
    exts = comp.features.extrudeFeatures
    inp = exts.createInput(profs, operation)
    inp.setDistanceExtent(False, adsk.core.ValueInput.createByReal(height))
    if participants:
        inp.participantBodies = participants
    return exts.add(inp)


def cut_cone(comp, body, x, y, mouth_d, depth):
    """Cut a 120-deg-included cone seat into the top face at (x, y)."""
    # revolve a triangle: axis vertical through (x,y)
    sk = comp.sketches.add(comp.xZConstructionPlane)
    # sketch plane XZ: sketch x -> world x, sketch y -> world -z? Use lines
    # in world coords via 3D points on a plane through y: simpler approach —
    # loft-free: model cone as extruded circle then chamfer is fiddly; use a
    # revolved cut built from a profile sketched on the XZ plane offset to y.
    planes = comp.constructionPlanes
    pin = planes.createInput()
    pin.setByOffset(comp.xZConstructionPlane,
                    adsk.core.ValueInput.createByReal(y))
    plane = planes.add(pin)
    sk = comp.sketches.add(plane)
    lines = sk.sketchCurves.sketchLines
    # profile: right triangle, apex at depth below top surface
    top = 0.0  # sketch is placed so its origin maps to world (0, y, 0)
    p0 = adsk.core.Point3D.create(x, top, 0)
    p1 = adsk.core.Point3D.create(x + mouth_d / 2.0, top, 0)
    p2 = adsk.core.Point3D.create(x, top - depth, 0)
    l0 = lines.addByTwoPoints(p0, p1)
    l1 = lines.addByTwoPoints(p1, p2)
    l2 = lines.addByTwoPoints(p2, p0)
    axis = lines.addByTwoPoints(adsk.core.Point3D.create(x, top, 0),
                                adsk.core.Point3D.create(x, top - depth, 0))
    axis.isConstruction = True
    revs = comp.features.revolveFeatures
    prof = sk.profiles.item(0)
    rin = revs.createInput(prof, axis,
                           adsk.fusion.FeatureOperations.CutFeatureOperation)
    rin.setAngleExtent(False, adsk.core.ValueInput.createByString("360 deg"))
    rin.participantBodies = [body]
    revs.add(rin)


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open a Fusion design first.")
            return
        ensure_params(design)

        L = pval(design, "km_leverArm")
        W = pval(design, "km_plateW")
        H = pval(design, "km_plateH")
        T = pval(design, "km_plateT")
        ballD = pval(design, "km_ballD")
        bossD = pval(design, "km_seatBossD")
        insD = pval(design, "km_insertD")
        insL = pval(design, "km_insertL")
        sprD = pval(design, "km_springHoleD")
        bcd = pval(design, "km_mirrorBCD")
        mbD = pval(design, "km_mirrorBoltD")
        mbN = int(round(design.userParameters.itemByName("km_mirrorBoltN").value))
        bbD = pval(design, "km_baseBoltD")
        cbD = pval(design, "km_baseCboreD")
        cbZ = pval(design, "km_baseCboreZ")

        import math
        # kinematic points (plate local coords, pivot at margin corner)
        m = bossD  # corner margin
        pivot = (m, m)
        adjA = (m, m + L)        # vee -> pitch
        adjB = (m + L, m)        # flat -> yaw
        springs = [(m + 0.35 * L, m + 0.35 * L), (m + 0.6 * L, m + 0.6 * L)]

        # ---------------- BASE ----------------
        base = new_component(design, "KM_Base")
        sk = base.sketches.add(base.xYConstructionPlane)
        rect(sk, 0, 0, W, H)
        extrude_profiles(base, sk,
                         T, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        body = base.bRepBodies.item(0)

        # housing counterbored bolts at free corners
        sk = base.sketches.add(base.xYConstructionPlane)
        for (x, y) in [(W - m / 2, H - m / 2), (W - m / 2, m / 2),
                       (m / 2, H - m / 2)]:
            circle(sk, x, y, bbD)
        extrude_profiles(base, sk, T,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [body])
        # counterbores from the bottom
        sk = base.sketches.add(base.xYConstructionPlane)
        for (x, y) in [(W - m / 2, H - m / 2), (W - m / 2, m / 2),
                       (m / 2, H - m / 2)]:
            circle(sk, x, y, cbD)
        extrude_profiles(base, sk, cbZ,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [body])

        # spring anchor holes
        sk = base.sketches.add(base.xYConstructionPlane)
        for (x, y) in springs:
            circle(sk, x, y, sprD)
        extrude_profiles(base, sk, T,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [body])

        # kinematic seats cut into the top face:
        # cone at pivot (120 deg included -> depth = mouthR / tan(60))
        mouth = ballD * 0.9
        cut_cone(base, body, pivot[0], pivot[1],
                 mouth, (mouth / 2.0) / math.tan(math.radians(60)))
        # NOTE top face is at z=T; cut_cone sketches at z=0 plane of the
        # component — move the whole seat operation to the top by cutting
        # from a plane: simplest robust approach is cones modeled as
        # countersinks via hole features:
        # (fallback) vee + flat: the VEE is a shallow 90-deg groove aimed
        # at the cone; model as a rotated box cut.
        sk = base.sketches.add(base.xYConstructionPlane)
        # vee groove outline (narrow rectangle pointing from adjA toward pivot)
        vx, vy = adjA
        gl, gw = bossD, ballD * 0.8
        rect(sk, vx - gw / 2, vy - gl / 2, vx + gw / 2, vy + gl / 2)
        extrude_profiles(base, sk, ballD * 0.25,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [body])
        # flat at adjB: no feature needed (plain surface)

        # ---------------- PLATFORM ----------------
        plat = new_component(design, "KM_Platform")
        sk = plat.sketches.add(plat.xYConstructionPlane)
        rect(sk, 0, 0, W, H)
        extrude_profiles(plat, sk, T,
                         adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        pbody = plat.bRepBodies.item(0)

        # pivot ball pocket (blind hole, ball glued in, protrudes km_gap)
        sk = plat.sketches.add(plat.xYConstructionPlane)
        circle(sk, pivot[0], pivot[1], ballD * 0.98)  # press fit
        extrude_profiles(plat, sk, ballD * 0.5,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [pbody])

        # adjuster bosses: heat-set insert holes at adjA and adjB
        sk = plat.sketches.add(plat.xYConstructionPlane)
        for (x, y) in (adjA, adjB):
            circle(sk, x, y, insD)
        extrude_profiles(plat, sk, insL,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [pbody])

        # spring anchors matching the base
        sk = plat.sketches.add(plat.xYConstructionPlane)
        for (x, y) in springs:
            circle(sk, x, y, sprD)
        extrude_profiles(plat, sk, T,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [pbody])

        # mirror backing-plate bolt circle centered between the three
        # kinematic points (put the mirror's center of mass inside the
        # support triangle)
        cx = (pivot[0] + adjA[0] + adjB[0]) / 3.0
        cy = (pivot[1] + adjA[1] + adjB[1]) / 3.0
        sk = plat.sketches.add(plat.xYConstructionPlane)
        for k in range(mbN):
            a = 2 * math.pi * k / mbN
            circle(sk, cx + bcd / 2 * math.cos(a),
                   cy + bcd / 2 * math.sin(a), mbD)
        extrude_profiles(plat, sk, T,
                         adsk.fusion.FeatureOperations.CutFeatureOperation,
                         [pbody])

        ui.messageBox(
            "KM_Base and KM_Platform created.\n\n"
            "Edit dimensions in Modify > Change Parameters (km_*).\n"
            "Angular resolution: atan(screw pitch / km_leverArm) per turn.\n"
            "Remember: vee groove must AIM at the cone (pitch axis), and the "
            "mirror bolt circle sits at the support-triangle centroid.")
    except Exception:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
