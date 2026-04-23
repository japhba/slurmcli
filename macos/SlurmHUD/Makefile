PROJECT := SlurmHUD.xcodeproj
SCHEME := SlurmHUD
DERIVED_DATA := /tmp/slurmhud-derived
APP_PATH := $(DERIVED_DATA)/Build/Products/Debug/SlurmHUD.app
APPEX_PATH := $(APP_PATH)/Contents/PlugIns/SlurmHUDWidget.appex
APP_ENTITLEMENTS := codesign/app.entitlements
WIDGET_ENTITLEMENTS := codesign/widget.entitlements

.PHONY: generate build build-widget clean-derived rebuild sign verify-sign info

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
	$(MAKE) sign

build-widget:
	xcodebuild \
		-project $(PROJECT) \
		-scheme SlurmHUDWidget \
		-configuration Debug \
		-derivedDataPath $(DERIVED_DATA)-widget \
		CODE_SIGNING_ALLOWED=NO \
		CODE_SIGNING_REQUIRED=NO \
		build

# Re-sign the app and widget ad-hoc with the App Group entitlement embedded.
# Without this, macOS TCC prompts ("SlurmHUD would like to access data from
# other apps") every time a freshly-launched process touches the Group
# Container — and the LaunchAgent spawns a fresh process every refresh
# cycle. Ad-hoc signing (`-` identity) needs no Apple Developer account;
# it's purely about embedding the entitlement so TCC trusts the access.
sign:
	@test -d "$(APP_PATH)" || (echo "no built app at $(APP_PATH); run `make build` first"; exit 1)
	@test -d "$(APPEX_PATH)" || (echo "no widget appex at $(APPEX_PATH)"; exit 1)
	codesign --force --sign - --entitlements "$(WIDGET_ENTITLEMENTS)" --timestamp=none "$(APPEX_PATH)"
	codesign --force --sign - --entitlements "$(APP_ENTITLEMENTS)" --timestamp=none "$(APP_PATH)"
	@echo "Signed; verifying..."
	$(MAKE) verify-sign

verify-sign:
	@echo "--- App entitlements ---"
	codesign -d --entitlements - "$(APP_PATH)" 2>&1 | tail -n +2
	@echo "--- Widget entitlements ---"
	codesign -d --entitlements - "$(APPEX_PATH)" 2>&1 | tail -n +2

clean-derived:
	rm -rf $(DERIVED_DATA) $(DERIVED_DATA)-widget

rebuild: clean-derived generate build

info:
	@echo "App Info:"
	plutil -p $(APP_PATH)/Contents/Info.plist
	@echo
	@echo "Widget Info:"
	plutil -p $(APPEX_PATH)/Contents/Info.plist
