PROJECT := SlurmHUD.xcodeproj
SCHEME := SlurmHUD
DERIVED_DATA := /tmp/slurmhud-derived

.PHONY: generate build build-widget clean-derived rebuild info

generate:
	xcodegen generate --spec project.yml

build:
	xcodebuild \
		-project $(PROJECT) \
		-scheme $(SCHEME) \
		-configuration Debug \
		-derivedDataPath $(DERIVED_DATA) \
		CODE_SIGNING_ALLOWED=NO \
		CODE_SIGNING_REQUIRED=NO \
		build

build-widget:
	xcodebuild \
		-project $(PROJECT) \
		-scheme SlurmHUDWidget \
		-configuration Debug \
		-derivedDataPath $(DERIVED_DATA)-widget \
		CODE_SIGNING_ALLOWED=NO \
		CODE_SIGNING_REQUIRED=NO \
		build

clean-derived:
	rm -rf $(DERIVED_DATA) $(DERIVED_DATA)-widget

rebuild: clean-derived generate build

info:
	@echo "App Info:"
	plutil -p $(DERIVED_DATA)/Build/Products/Debug/SlurmHUD.app/Contents/Info.plist
	@echo
	@echo "Widget Info:"
	plutil -p $(DERIVED_DATA)/Build/Products/Debug/SlurmHUD.app/Contents/PlugIns/SlurmHUDWidget.appex/Contents/Info.plist
