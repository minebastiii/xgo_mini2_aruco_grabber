#!/usr/bin/env python3
"""
grasp_test.py — läuft auf dem ROBOT (Pi)

Eigenständiges Test-Skript, KEIN ROS2-Node. Bringt den Roboter in die
nach vorn gebeugte Haltung (wie am Anfang von ArucoFSM.__init__) und
führt danach exakt die GRASP-Substep-Sequenz aus fsm_node_new.py aus.

Kein Aruco-Tracking, keine Trigger — der Greifvorgang wird einfach
direkt ausgelöst, sobald das Skript startet.

Nutzung:
    python3 grasp_test.py
"""

import time
from xgolib import XGO


def main():
    xgo = XGO(port='/dev/ttyAMA0')

    print("[grasp_test] Stoppe und nehme Grundhaltung (vorn gebeugt) ein...")
    xgo.stop()
    xgo.translation("z", 80)
    xgo.attitude("p", 15)
    time.sleep(2.0)  # Zeit geben, die Pose einzunehmen

    print("[grasp_test] Starte GRASP-Sequenz...")

    # substep 0
    xgo.claw(0)
    time.sleep(1.0)

    # substep 1
    xgo.move("x", 2.5)
    time.sleep(1.0)

    # substep 2
    xgo.stop()
    time.sleep(1.0)

    # # substep 3
    # #xgo.arm_motor([-25, 90, 0])
    # xgo.arm(-25, 90)
    # time.sleep(1.0)
    #
    # # substep 4
    # xgo.claw(255)
    # time.sleep(2.0)
    #
    # # substep 5
    # #xgo.arm_motor([20, -90, 0])
    # xgo.arm(20, -90)
    # time.sleep(1.0)

    # substep 6
    #xgo.arm_motor([83, -90, 0])
    xgo.arm(83, -90)
    time.sleep(1.0)

    xgo.claw(255)
    time.sleep(2.0)

    xgo.reset()

    angles = xgo.read_motor()
    print(angles[12:15])

    # xgo.arm(0, 75)
    # time.sleep(1.0)
    #
    # xgo.arm(0, 50)
    # time.sleep(1.0)

    print("[grasp_test] GRASP-Sequenz abgeschlossen.")
    xgo.stop()


if __name__ == '__main__':
    main()
