PROJECT := SlurmHUD.xcodeproj
SCHEME := SlurmHUD
DERIVED_DATA := /tmp/slurmhud-derived
APP_ENTITLEMENTS := codesign/app.entitlements
WIDGET_ENTITLEMENTS := codesign/widget.entitlements

# Ad-hoc signing. See project.yml for why we don't use Apple Development.
CODE_SIGN_IDENTITY ?= -

# Build straight into ~/Applications so Spotlight indexes it and chronod
# (the widget host) can discover the embedded widget. Apps under /private/tmp
# are invisible to both. xcodebuild's intermediates still go to DERIVED_DATA.
INSTALL_DIR := $(HOME)/Applications
APP_PATH := $(INSTALL_DIR)/SlurmHUD.app
APPEX_PATH := $(APP_PATH)/Contents/PlugIns/SlurmHUDWidget.appex

.PHONY: generate build clean-derived rebuild sign verify-sign info uninstall

generate:
	xcodegen generate --spec project.yml

# XCODEBUILD_FLAGS lets callers override project.yml settings on the command
# line. CI uses it to swap Apple Development signing for ad-hoc; locally,
# the empty default lets the project.yml settings flow through.
XCODEBUILD_FLAGS ?=

build:
	mkdir -p "$(INSTALL_DIR)"
	rm -rf "$(APP_PATH)"
	xcodebuild \
		-project $(PROJECT) \
		-scheme $(SCHEME) \
		-configuration Debug \
		-derivedDataPath $(DERIVED_DATA) \
		CONFIGURATION_BUILD_DIR="$(INSTALL_DIR)" \
		$(XCODEBUILD_FLAGS) \
		build
	$(MAKE) sign
	# Re-register with LaunchServices so Spotlight/chronod see the new
	# bundle and drop stale entries from prior build paths.
	/System/Library/Frameworks/CoreServices.framework/Versions/Current/Frameworks/LaunchServices.framework/Versions/Current/Support/lsregister -f -R -trusted "$(APP_PATH)"
	@echo "Built and installed to $(APP_PATH)"

# Re-sign the app and widget with a stable identity and the App Group
# entitlement embedded. Without a stable signature, macOS TCC prompts
# ("SlurmHUD would like to access data from other apps") on every rebuild
# because the code identity changes. CODE_SIGN_IDENTITY defaults to a local
# Apple Development cert; pass `=-` to fall back to ad-hoc.
sign:
	@test -d "$(APP_PATH)" || (echo "no built app at $(APP_PATH); run 'make build' first"; exit 1)
	@test -d "$(APPEX_PATH)" || (echo "no widget appex at $(APPEX_PATH)"; exit 1)
	codesign --force --sign "$(CODE_SIGN_IDENTITY)" --entitlements "$(WIDGET_ENTITLEMENTS)" --timestamp=none "$(APPEX_PATH)"
	codesign --force --sign "$(CODE_SIGN_IDENTITY)" --entitlements "$(APP_ENTITLEMENTS)" --timestamp=none "$(APP_PATH)"
	@echo "Signed; verifying..."
	$(MAKE) verify-sign

verify-sign:
	@echo "--- App entitlements ---"
	codesign -d --entitlements - "$(APP_PATH)" 2>&1 | tail -n +2
	@echo "--- Widget entitlements ---"
	codesign -d --entitlements - "$(APPEX_PATH)" 2>&1 | tail -n +2

clean-derived:
	rm -rf $(DERIVED_DATA)

rebuild: clean-derived generate build

uninstall:
	rm -rf "$(APP_PATH)"
	/System/Library/Frameworks/CoreServices.framework/Versions/Current/Frameworks/LaunchServices.framework/Versions/Current/Support/lsregister -u "$(APP_PATH)" 2>/dev/null || true

info:
	@echo "App Info:"
	plutil -p $(APP_PATH)/Contents/Info.plist
	@echo
	@echo "Widget Info:"
	plutil -p $(APPEX_PATH)/Contents/Info.plist
