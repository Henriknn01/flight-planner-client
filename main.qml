import QtQuick 6.5
import QtQuick.Controls 6.5
import QtQuick.Dialogs 6.5
import QtQuick.Scene3D
import Qt3D.Core
import Qt3D.Render
import Qt3D.Input
import Qt3D.Extras

Rectangle {
    id: rectangle
    width: 512
    height: 1024
    visible: true
    color: "#ffffff"

    FileDialog {
        id: fileDialog
        title: "please choose a file"
        onAccepted: {
            ModelUpload.read_file(fileDialog.fileUrls)
        }
    }

    Rectangle {
        id: controlPanel
        x: 1475
        y: 45
        width: 400
        opacity: 1
        visible: true
        color: "#ffffff"
        radius: 20
        border.width: 0
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        anchors.leftMargin: 45
        anchors.rightMargin: 45
        anchors.topMargin: 45
        anchors.bottomMargin: 45

        Column {
            id: column
            visible: true
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: parent.top
            anchors.bottom: parent.bottom
            anchors.leftMargin: 21
            anchors.rightMargin: 20
            anchors.topMargin: 20
            anchors.bottomMargin: 20
            spacing: 0

            Item {
                id: vesselInfo
                height: vesselInfoCol.implicitHeight
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: 0
                anchors.rightMargin: 0

                Column {
                    id: vesselInfoCol
                    height: vesselLabel.height + vesselName.height
                    visible: true
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.leftMargin: 0
                    anchors.rightMargin: 0

                    Text {
                        id: vesselLabel
                        text: qsTr("Vessel")
                        font.pixelSize: 18
                    }

                    Text {
                        id: vesselName
                        text: qsTr(Backend.vessel_name)
                        font.pixelSize: 24
                        font.styleName: "Bold"
                    }

                    Button {
                        id: selectShip
                        text: qsTr("select ship model")
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 0
                        anchors.rightMargin: 0
                        flat: false
                        highlighted: true
                        checkable: false
                        onPressed: {
                            fileDialog.open()
                        }
                    }
                }
            }

            Item {
                id: userInput
                height: userInputCol.implicitHeight
                visible: true
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: vesselInfo.bottom
                anchors.leftMargin: 0
                anchors.rightMargin: 0
                anchors.topMargin: 20

                Column {
                    id: userInputCol
                    height: userInputCol.implicitHeight
                    anchors.left: parent.left
                    anchors.right: parent.right
                    anchors.leftMargin: 0
                    anchors.rightMargin: 0

                    Text {
                        id: inputLabel
                        text: qsTr("Input")
                        font.pixelSize: 12
                        font.styleName: "Light"
                    }
                    Text {
                        id: waypointOffsetLabel
                        text: qsTr("Waypoint Offset")
                        font.pixelSize: 18
                    }

                    Item {
                        id: row
                        height: waypointOffsetSlider.height
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 0
                        anchors.rightMargin: 0

                        Slider {
                            id: waypointOffsetSlider
                            value: 50
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.leftMargin: 0
                            anchors.rightMargin: 39
                            anchors.topMargin: 0
                            stepSize: 1
                            snapMode: RangeSlider.SnapAlways
                            to: 100
                        }

                        Text {
                            id: waypointOffsetValueLabel
                            text: waypointOffsetSlider.value
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.right: parent.right
                            anchors.rightMargin: 0
                            font.pixelSize: 16
                            horizontalAlignment: Text.AlignLeft
                            font.styleName: "Light"
                            rightPadding: 10
                            leftPadding: 10
                            padding: 0
                        }
                    }

                    Text {
                        id: overlapAmountLabel
                        text: qsTr("Overlap Amount")
                        font.pixelSize: 18
                    }

                    Item {
                        id: row3
                        height: overlapAmountSlider.height
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 0
                        anchors.rightMargin: 0

                        Slider {
                            id: overlapAmountSlider
                            value: 0.5
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.leftMargin: 0
                            anchors.rightMargin: 42
                            stepSize: 0.1
                            snapMode: RangeSlider.SnapAlways
                        }

                        Text {
                            id: text2
                            text: overlapAmountSlider.value.toFixed(1)
                            anchors.verticalCenter: overlapAmountSlider.verticalCenter
                            anchors.right: parent.right
                            anchors.rightMargin: 0
                            font.pixelSize: 16
                            rightPadding: 10
                            leftPadding: 10
                            font.styleName: "Light"
                        }
                    }

                    Text {
                        id: altitudeRangeLabel
                        text: qsTr("Altitude Range")
                        font.pixelSize: 18
                    }

                    Item {
                        id: row4
                        height: rangeSliderAltitude.implicitHeight
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 0
                        anchors.rightMargin: 0

                        Text {
                            id: minRangeSliderAltitude
                            text: rangeSliderAltitude.first.value
                            anchors.verticalCenter: rangeSliderAltitude.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: 0
                            font.pixelSize: 16
                            font.styleName: "Light"
                            rightPadding: 10
                            leftPadding: 10
                        }

                        RangeSlider {
                            id: rangeSliderAltitude
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.leftMargin: 42
                            anchors.rightMargin: 42
                            to: 15
                            snapMode: RangeSlider.SnapAlways
                            stepSize: 0.5
                            second.value: 7
                            first.value: 2.5
                        }

                        Text {
                            id: maxRangeSliderAltitude
                            text: rangeSliderAltitude.second.value
                            anchors.verticalCenter: rangeSliderAltitude.verticalCenter
                            anchors.right: parent.right
                            anchors.rightMargin: 0
                            font.pixelSize: 16
                            font.styleName: "Light"
                            rightPadding: 10
                            leftPadding: 10
                        }
                    }

                    Text {
                        id: scanRangeLabel
                        text: qsTr("Scan Range")
                        font.pixelSize: 18
                    }

                    Item {
                        id: row5
                        height: rangeSliderScan.implicitHeight
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.leftMargin: 0
                        anchors.rightMargin: 0
                        Text {
                            id: minRangeSliderScan
                            text: rangeSliderScan.first.value
                            anchors.verticalCenter: rangeSliderScan.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: 0
                            font.pixelSize: 16
                            rightPadding: 10
                            leftPadding: 10
                            font.styleName: "Light"
                        }

                        RangeSlider {
                            id: rangeSliderScan
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.leftMargin: 42
                            anchors.rightMargin: 42
                            stepSize: 0.5
                            snapMode: RangeSlider.SnapAlways
                            second.value: 7
                            first.value: 0
                            to: 15
                        }

                        Text {
                            id: maxRangeSliderScan
                            text: rangeSliderScan.second.value
                            anchors.verticalCenter: rangeSliderScan.verticalCenter
                            anchors.right: parent.right
                            anchors.rightMargin: 0
                            font.pixelSize: 16
                            rightPadding: 10
                            leftPadding: 10
                            font.styleName: "Light"
                        }
                    }
                }
            }

            Button {
                id: button
                text: qsTr("Generate Path")
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.leftMargin: 0
                anchors.rightMargin: 0
                anchors.bottomMargin: 0
                autoExclusive: false
                highlighted: true
                onPressed: Backend.update_vessel_name()
            }
        }
    }
}