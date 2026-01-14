class Slurmhud < Formula
  desc "SlurmHUD macOS menu bar app + widget"
  homepage "https://github.com/your-org/slurmcli"
  url "https://github.com/your-org/slurmcli/archive/refs/heads/main.tar.gz"
  version "0.1.0"
  sha256 ""

  depends_on "xcodegen" => :build

  def install
    cd "macos/SlurmHUD" do
      system "xcodegen", "generate", "--spec", "project.yml"
      system "xcodebuild", "-project", "SlurmHUD.xcodeproj",
             "-scheme", "SlurmHUD", "-configuration", "Release",
             "CONFIGURATION_BUILD_DIR=#{buildpath}/build"
      prefix.install "build/SlurmHUD.app"
    end
  end

  def caveats
    <<~EOS
      SlurmHUD.app installed to:
        #{prefix}/SlurmHUD.app

      You may want to copy it into /Applications.
    EOS
  end
end
